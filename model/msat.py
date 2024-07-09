from inspect import stack
import torch
#torch.manual_seed(0)
import random
import os
import numpy as np
from torch import nn
from torch.nn import functional as F
from collections import defaultdict
from .common.mlp import MLP
from .msat_loss import loss_func
from .common.dist import *
from .msat_lib import msatEncoderLayer, msatDecoderLayer, msatDecoder, msatEncoder
from .map_encoder import MapEncoder
from utils.torch import *
from utils.utils import initialize_weights
from scipy import spatial
from .fuse_sensor import FuseModule
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import cm 
from .fuse_network import Transformer

#TODO: remove tensor2array after done with figures
def tensor2array(tensor, max_value=None, colormap='bone'):
    #tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.abs().max()  # .item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:

        norm_array = tensor.squeeze()/max_value
        norm_array = norm_array.detach().cpu().numpy()

        array = cm.get_cmap(colormap)(norm_array).astype(np.float32)

        if tensor.ndimension() == 2 and tensor.size(0) == 1:
            array=np.expand_dims(array,axis=0)
       
        array = array.transpose(2, 0, 1)
    return array

#from fusionnetwork import PoseFusionNet
from .fuse_network import Transformer
from .FullNet import FullNet
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#setup_seed(0)

def merge_tensors(lidar,radar): 
    """
    # torch stack
    # N,X,32
    # N is the number of merged sensors, X is the batch size, 32 is out dim.
    """
    tensor = torch.stack((lidar,radar))
    # tensor = torch.flatten(tensor)

    return tensor # merge sensor tensors and flatten them.

def init_models(cfg, sensor):
    
    if sensor == 'radar' or 'lidar':
        model = MapEncoder(cfg.map_encoder, ful_con = 196)
        
    if sensor == 'camera':
        model = MapEncoder(cfg.map_encoder, ful_con = 32)
    
    return model

def generate_ar_mask(sz, agent_num, agent_mask):
    assert sz % agent_num == 0
    T = sz // agent_num
    mask = agent_mask.repeat(T, T)
    for t in range(T-1):
        i1 = t * agent_num
        i2 = (t+1) * agent_num
        mask[i1:i2, i2:] = float('-inf')
    return mask


def generate_mask(tgt_sz, src_sz, agent_num, agent_mask):
    assert tgt_sz % agent_num == 0 and src_sz % agent_num == 0
    mask = agent_mask.repeat(tgt_sz // agent_num, src_sz // agent_num)
    return mask


""" Positional Encoding """
class PositionalAgentEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_t_len=200, max_a_len=200, concat=False, use_agent_enc=False, agent_enc_learn=False):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        self.use_agent_enc = use_agent_enc
        if concat:
            self.fc = nn.Linear((3 if use_agent_enc else 2) * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)
        if use_agent_enc:
            if agent_enc_learn:
                self.ae = nn.Parameter(torch.randn(max_a_len, 1, d_model) * 0.1)
            else:
                ae = self.build_pos_enc(max_a_len)
                self.register_buffer('ae', ae)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def build_agent_enc(self, max_len):
        ae = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        ae[:, 0::2] = torch.sin(position * div_term)
        ae[:, 1::2] = torch.cos(position * div_term)
        ae = ae.unsqueeze(0).transpose(0, 1)
        return ae
    
    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe.repeat_interleave(num_a, dim=0)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset, agent_enc_shuffle):
        if agent_enc_shuffle is None:
            ae = self.ae[a_offset: num_a + a_offset, :]
        else:
            ae = self.ae[agent_enc_shuffle]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, agent_enc_shuffle=None, t_offset=0, a_offset=0):
        num_t = x.shape[0] // num_a
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset)
        if self.use_agent_enc:
            agent_enc = self.get_agent_enc(num_t, num_a, a_offset, agent_enc_shuffle)
        if self.concat:
            feat = [x, pos_enc.repeat(1, x.size(1), 1)]
            if self.use_agent_enc:
                feat.append(agent_enc.repeat(1, x.size(1), 1))
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
            if self.use_agent_enc:
                x += agent_enc
        return self.dropout(x)


""" Context (Past) Encoder """
class ContextEncoder(nn.Module):
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.ctx = ctx
        self.motion_dim = ctx['motion_dim']
        self.model_dim = ctx['tf_model_dim']
        self.ff_dim = ctx['tf_ff_dim']
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = cfg.get('nlayer', 6)
        self.input_type = ctx['input_type']
        self.pooling = cfg.get('pooling', 'mean')
        self.agent_enc_shuffle = ctx['agent_enc_shuffle']
        self.vel_heading = ctx['vel_heading']
        ctx['context_dim'] = self.model_dim
        in_dim = self.motion_dim * len(self.input_type)
        
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - self.motion_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        encoder_layers = msatEncoderLayer(ctx['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_encoder = msatEncoder(encoder_layers, self.nlayer)
        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=ctx['pos_concat'], max_a_len=ctx['max_agent_len'], use_agent_enc=ctx['use_agent_enc'], agent_enc_learn=ctx['agent_enc_learn'])

    def forward(self, data):
        traj_in = []
        for key in self.input_type:
            if key == 'pos':
                traj_in.append(data['pre_motion'])
            elif key == 'vel':
                vel = data['pre_vel']
                if len(self.input_type) > 1:
                    vel = torch.cat([vel[[0]], vel], dim=0)
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data['heading'])[0]
                #vel=vel.unsqueeze(0).expand((data['map_enc'].shape[0],-1, -1, -1))
                traj_in.append(vel)
            elif key == 'norm':
                #data['pre_motion_norm']=data['pre_motion_norm'].unsqueeze(0).expand((data['map_enc'].shape[0],-1, -1, -1))
                traj_in.append(data['pre_motion_norm']) 
            elif key == 'scene_norm':
             
                #data['pre_motion_scene_norm']=data['pre_motion_scene_norm'].unsqueeze(0).expand((data['map_enc'].shape[0],-1, -1, -1))
                traj_in.append(data['pre_motion_scene_norm'])
                #print('pre_motion_scene_norm',data['pre_motion_scene_norm'].shape)            
            elif key == 'heading':
                #hv = data['heading_vec'].unsqueeze(0).repeat((data['map_enc'].shape[0],data['pre_motion'].shape[0], 1, 1))
                hv = data['heading_vec'].unsqueeze(0).repeat((data['pre_motion'].shape[0], 1, 1))
                traj_in.append(hv)
            elif key == 'map':
                #map_enc = data['map_enc'].unsqueeze(1).expand(-1,data['pre_motion'].shape[0],-1,-1)
                map_enc = data['map_enc'].unsqueeze(0).repeat((data['pre_motion'].shape[0], 1, 1))
                traj_in.append(map_enc)
                 
            else:
                raise ValueError('unknown input_type!')
        traj_in = torch.cat(traj_in, dim=-1)
        tf_in = self.input_fc(traj_in.view(-1, traj_in.shape[-1])).view(-1, 1, self.model_dim)
        agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
        tf_in_pos = self.pos_encoder(tf_in, num_a=data['agent_num'], agent_enc_shuffle=agent_enc_shuffle)
        
        src_agent_mask = data['agent_mask'].clone()
        src_mask = generate_mask(tf_in.shape[0], tf_in.shape[0], data['agent_num'], src_agent_mask).to(tf_in.device)
        
        data['context_enc'] = self.tf_encoder(tf_in_pos, mask=src_mask, num_agent=data['agent_num'])
        
        context_rs = data['context_enc'].view(-1, data['agent_num'], self.model_dim)
        # compute per agent context
        if self.pooling == 'mean':
            data['agent_context'] = torch.mean(context_rs, dim=0)
        else:
            data['agent_context'] = torch.max(context_rs, dim=0)[0]


""" Future Encoder """
class FutureEncoder(nn.Module):
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.context_dim = context_dim = ctx['context_dim']
        self.forecast_dim = forecast_dim = ctx['forecast_dim']
        self.nz = ctx['nz']
        self.z_type = ctx['z_type']
        self.z_tau_annealer = ctx.get('z_tau_annealer', None)
        self.model_dim = ctx['tf_model_dim']
        self.ff_dim = ctx['tf_ff_dim']
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = cfg.get('nlayer', 6)
        self.out_mlp_dim = cfg.get('out_mlp_dim', None)
        self.input_type = ctx['fut_input_type']
        self.pooling = cfg.get('pooling', 'mean')
        self.agent_enc_shuffle = ctx['agent_enc_shuffle']
        self.vel_heading = ctx['vel_heading']
        # networks
        in_dim = forecast_dim * len(self.input_type)
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - forecast_dim    
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        decoder_layers = msatDecoderLayer(ctx['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_decoder = msatDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=ctx['pos_concat'], max_a_len=ctx['max_agent_len'], use_agent_enc=ctx['use_agent_enc'], agent_enc_learn=ctx['agent_enc_learn'])
        num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
        if self.out_mlp_dim is None:
            self.q_z_net = nn.Linear(self.model_dim, num_dist_params)
        else:
            self.out_mlp = MLP(self.model_dim, self.out_mlp_dim, 'relu')
            self.q_z_net = nn.Linear(self.out_mlp.out_dim, num_dist_params)
        # initialize
        initialize_weights(self.q_z_net.modules())

    def forward(self, data, reparam=True):
        traj_in = []
        for key in self.input_type:
            if key == 'pos':
                traj_in.append(data['fut_motion'])
            elif key == 'vel':
                vel = data['fut_vel']
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data['heading'])[0]
                #vel=vel.unsqueeze(0).expand((data['map_enc'].shape[0],-1, -1, -1))
                traj_in.append(vel)
            elif key == 'norm':
                traj_in.append(data['fut_motion_norm'])
            elif key == 'scene_norm':
                #data['fut_motion_scene_norm']= data['fut_motion_scene_norm'].unsqueeze(0).expand((data['map_enc'].shape[0],-1, -1, -1))
                traj_in.append(data['fut_motion_scene_norm'])
            elif key == 'heading':
                #hv = data['heading_vec'].unsqueeze(0).repeat((data['map_enc'].shape[0],data['fut_motion'].shape[0], 1, 1))
                hv = data['heading_vec'].unsqueeze(0).repeat((data['fut_motion'].shape[0], 1, 1))
                traj_in.append(hv)
            elif key == 'map':

                #map_enc = data['map_enc'].unsqueeze(1).expand(-1,data['fut_motion'].shape[0],-1,-1)
                map_enc = data['map_enc'].unsqueeze(0).repeat((data['fut_motion'].shape[0], 1, 1))
                traj_in.append(map_enc)
            else:
                raise ValueError('unknown input_type!')
        traj_in = torch.cat(traj_in, dim=-1)   
        tf_in = self.input_fc(traj_in.view(-1, traj_in.shape[-1])).view(-1, 1, self.model_dim)
        agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
        tf_in_pos = self.pos_encoder(tf_in, num_a=data['agent_num'], agent_enc_shuffle=agent_enc_shuffle)

        mem_agent_mask = data['agent_mask'].clone()
        tgt_agent_mask = data['agent_mask'].clone()
        mem_mask = generate_mask(tf_in.shape[0], data['context_enc'].shape[0], data['agent_num'], mem_agent_mask).to(tf_in.device)
        tgt_mask = generate_mask(tf_in.shape[0], tf_in.shape[0], data['agent_num'], tgt_agent_mask).to(tf_in.device)
        
        #TODO Modify tf_out and fix q_z_params, make sure data['q_z_samp'] returns correct.

        tf_out, _ = self.tf_decoder(tf_in_pos, data['context_enc'], memory_mask=mem_mask, tgt_mask=tgt_mask, num_agent=data['agent_num'])
        tf_out = tf_out.view(traj_in.shape[0], -1, self.model_dim)

        if self.pooling == 'mean':
            h = torch.mean(tf_out, dim=0)           
        else:
            h = torch.max(tf_out, dim=0)[0]     
        if self.out_mlp_dim is not None:
            h = self.out_mlp(h)
        q_z_params = self.q_z_net(h)
        if self.z_type == 'gaussian':
            data['q_z_dist'] = Normal(params=q_z_params)
        else:
            data['q_z_dist'] = Categorical(logits=q_z_params, temp=self.z_tau_annealer.val())
        data['q_z_samp'] = data['q_z_dist'].rsample()


""" Future Decoder """
class FutureDecoder(nn.Module):
    def __init__(self, cfg, ctx, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.ar_detach = ctx['ar_detach']
        self.context_dim = context_dim = ctx['context_dim']
        self.forecast_dim = forecast_dim = ctx['forecast_dim']
        self.pred_scale = cfg.get('pred_scale', 1.0)
        self.pred_type = ctx['pred_type']
        self.sn_out_type = ctx['sn_out_type']
        self.sn_out_heading = ctx['sn_out_heading']
        self.input_type = ctx['dec_input_type']
        self.future_frames = ctx['future_frames']
        self.past_frames = ctx['past_frames']
        self.nz = ctx['nz']
        self.z_type = ctx['z_type']
        self.model_dim = ctx['tf_model_dim']
        self.ff_dim = ctx['tf_ff_dim']
        self.nhead = ctx['tf_nhead']
        self.dropout = ctx['tf_dropout']
        self.nlayer = cfg.get('nlayer', 6)
        self.out_mlp_dim = cfg.get('out_mlp_dim', None)
        self.pos_offset = cfg.get('pos_offset', False)
        self.agent_enc_shuffle = ctx['agent_enc_shuffle']
        self.learn_prior = ctx['learn_prior']
        # networks
        in_dim = forecast_dim + len(self.input_type) * forecast_dim + self.nz
        if 'map' in self.input_type:
            in_dim += ctx['map_enc_dim'] - forecast_dim 
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        decoder_layers = msatDecoderLayer(ctx['tf_cfg'], self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_decoder = msatDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=ctx['pos_concat'], max_a_len=ctx['max_agent_len'], use_agent_enc=ctx['use_agent_enc'], agent_enc_learn=ctx['agent_enc_learn'])
        if self.out_mlp_dim is None:
            self.out_fc = nn.Linear(self.model_dim, forecast_dim)
        else:
            in_dim = self.model_dim
            self.out_mlp = MLP(in_dim, self.out_mlp_dim, 'relu')
            self.out_fc = nn.Linear(self.out_mlp.out_dim, forecast_dim)
        initialize_weights(self.out_fc.modules())
        if self.learn_prior:
            num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
            self.p_z_net = nn.Linear(self.model_dim, num_dist_params)
            initialize_weights(self.p_z_net.modules())

    def decode_traj_ar(self, data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num, need_weights=False):
        agent_num = data['agent_num']
        if self.pred_type == 'vel':
            dec_in = pre_vel[[-1]]
        elif self.pred_type == 'pos':
            dec_in = pre_motion[[-1]]
        elif self.pred_type == 'scene_norm':
            dec_in = pre_motion_scene_norm[[-1]]      
        else:
            dec_in = torch.zeros_like(pre_motion[[-1]])
        dec_in = dec_in.view(-1, sample_num, dec_in.shape[-1])
        z_in = z.view(-1, sample_num, z.shape[-1])  
        in_arr = [dec_in, z_in]
    
        for key in self.input_type:
            if key == 'heading':

                heading = data['heading_vec'].unsqueeze(1).repeat((1, sample_num, 1))
                in_arr.append(heading)
            elif key == 'map':

                map_enc = data['map_enc'].unsqueeze(1).repeat((1, sample_num, 1))
                                                                                                                                                                                                  
                in_arr.append(map_enc)
            else:
                raise ValueError('wrong decoder input type!')

        dec_in_z = torch.cat(in_arr, dim=-1)
      
        
        mem_agent_mask = data['agent_mask'].clone()
        tgt_agent_mask = data['agent_mask'].clone()

        for i in range(self.future_frames):
            tf_in = self.input_fc(dec_in_z.view(-1, dec_in_z.shape[-1])).view(dec_in_z.shape[0], -1, self.model_dim)
            agent_enc_shuffle = data['agent_enc_shuffle'] if self.agent_enc_shuffle else None
            tf_in_pos = self.pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle, t_offset=self.past_frames-1 if self.pos_offset else 0)
            # tf_in_pos = tf_in
            mem_mask = generate_mask(tf_in.shape[0], context.shape[0], data['agent_num'], mem_agent_mask).to(tf_in.device)
            tgt_mask = generate_ar_mask(tf_in_pos.shape[0], agent_num, tgt_agent_mask).to(tf_in.device)

            tf_out, attn_weights = self.tf_decoder(tf_in_pos, context, memory_mask=mem_mask, tgt_mask=tgt_mask, num_agent=data['agent_num'], need_weights=need_weights)

            out_tmp = tf_out.view(-1, tf_out.shape[-1])
            if self.out_mlp_dim is not None:
                out_tmp = self.out_mlp(out_tmp)
            seq_out = self.out_fc(out_tmp).view(tf_out.shape[0], -1, self.forecast_dim)
            if self.pred_type == 'scene_norm' and self.sn_out_type in {'vel', 'norm'}:
                norm_motion = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
                if self.sn_out_type == 'vel':
                    norm_motion = torch.cumsum(norm_motion, dim=0)
                if self.sn_out_heading:
                    angles = data['heading'].repeat_interleave(sample_num)
                    norm_motion = rotation_2d_torch(norm_motion, angles)[0]
                seq_out = norm_motion + pre_motion_scene_norm[[-1]]
                seq_out = seq_out.view(tf_out.shape[0], -1, seq_out.shape[-1])
            if self.ar_detach:
                out_in = seq_out[-agent_num:].clone().detach()
            else:
                out_in = seq_out[-agent_num:]
            # create dec_in_z
            in_arr = [out_in, z_in]
            for key in self.input_type:
                if key == 'heading':
                    in_arr.append(heading)
                elif key == 'map':
                    in_arr.append(map_enc)
                else:
                    raise ValueError('wrong decoder input type!')
            out_in_z = torch.cat(in_arr, dim=-1)
            dec_in_z = torch.cat([dec_in_z, out_in_z], dim=0)

        seq_out = seq_out.view(-1, agent_num * sample_num, seq_out.shape[-1])
        data[f'{mode}_seq_out'] = seq_out

        if self.pred_type == 'vel':
            dec_motion = torch.cumsum(seq_out, dim=0)
            dec_motion += pre_motion[[-1]]
        elif self.pred_type == 'pos':
            dec_motion = seq_out.clone()
        elif self.pred_type == 'scene_norm':
            dec_motion = seq_out + data['scene_orig']
        else:
            dec_motion = seq_out + pre_motion[[-1]]

        dec_motion = dec_motion.transpose(0, 1).contiguous()       # M x frames x 7
        if mode == 'infer':
            dec_motion = dec_motion.view(-1, sample_num, *dec_motion.shape[1:])        # M x Samples x frames x 3
        data[f'{mode}_dec_motion'] = dec_motion
        if need_weights:
            data['attn_weights'] = attn_weights

    def decode_traj_batch(self, data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num):
        raise NotImplementedError

    def forward(self, data, mode, sample_num=1, autoregress=True, z=None, need_weights=False):
        context = data['context_enc'].repeat_interleave(sample_num, dim=1)       # 80 x 64
        pre_motion = data['pre_motion'].repeat_interleave(sample_num, dim=1)             # 10 x 80 x 2
        pre_vel = data['pre_vel'].repeat_interleave(sample_num, dim=1) if self.pred_type == 'vel' else None
        pre_motion_scene_norm = data['pre_motion_scene_norm'].repeat_interleave(sample_num, dim=1)
        
        # p(z)
        prior_key = 'p_z_dist' + ('_infer' if mode == 'infer' else '')
        if self.learn_prior:
            h = data['agent_context'].repeat_interleave(sample_num, dim=0)
            p_z_params = self.p_z_net(h)
            if self.z_type == 'gaussian':
                data[prior_key] = Normal(params=p_z_params)
            else:
                data[prior_key] = Categorical(params=p_z_params)
        else:
            if self.z_type == 'gaussian':
                data[prior_key] = Normal(mu=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device), logvar=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device))
            else:
                data[prior_key] = Categorical(logits=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device))

        if z is None:
            if mode in {'train', 'recon'}:
                z = data['q_z_samp'] if mode == 'train' else data['q_z_dist'].mode()
            elif mode == 'infer':
                z = data['p_z_dist_infer'].sample()
            else:
                raise ValueError('Unknown Mode!')

        if autoregress:
            self.decode_traj_ar(data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num, need_weights=need_weights)
        else:
            self.decode_traj_batch(data, mode, context, pre_motion, pre_vel, pre_motion_scene_norm, z, sample_num)
        
""" msat """
class msat(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        input_type = cfg.get('input_type', 'pos')
        pred_type = cfg.get('pred_type', input_type)
        if type(input_type) == str:
            input_type = [input_type]
        fut_input_type = cfg.get('fut_input_type', input_type)
        dec_input_type = cfg.get('dec_input_type', [])
        self.ctx = {
            'tf_cfg': cfg.get('tf_cfg', {}),
            'nz': cfg.nz,
            'z_type': cfg.get('z_type', 'gaussian'),
            'future_frames': cfg.future_frames,
            'past_frames': cfg.past_frames,
            'motion_dim': cfg.motion_dim,
            'forecast_dim': cfg.forecast_dim,
            'input_type': input_type,
            'fut_input_type': fut_input_type,
            'dec_input_type': dec_input_type,
            'pred_type': pred_type,
            'tf_nhead': cfg.tf_nhead,
            'tf_model_dim': cfg.tf_model_dim,
            'tf_ff_dim': cfg.tf_ff_dim,
            'tf_dropout': cfg.tf_dropout,
            'pos_concat': cfg.get('pos_concat', False),
            'ar_detach': cfg.get('ar_detach', True),
            'max_agent_len': cfg.get('max_agent_len', 128),
            'use_agent_enc': cfg.get('use_agent_enc', False),
            'agent_enc_learn': cfg.get('agent_enc_learn', False),
            'agent_enc_shuffle': cfg.get('agent_enc_shuffle', False),
            'sn_out_type': cfg.get('sn_out_type', 'scene_norm'),
            'sn_out_heading': cfg.get('sn_out_heading', False),
            'vel_heading': cfg.get('vel_heading', False),
            'learn_prior': cfg.get('learn_prior', False),
            'use_map': cfg.get('use_map', True)
        }
        self.sensor = cfg.get('sensor',None)
        self.use_map = self.ctx['use_map']
        self.rand_rot_scene = cfg.get('rand_rot_scene', False)
        self.discrete_rot = cfg.get('discrete_rot', False)
        self.map_global_rot = cfg.get('map_global_rot', False)
        self.ar_train = cfg.get('ar_train', True)
        self.max_train_agent = cfg.get('max_train_agent', 100)
        self.loss_cfg = self.cfg.loss_cfg
        self.loss_names = list(self.loss_cfg.keys())
        self.compute_sample = 'sample' in self.loss_names
        self.param_annealers = nn.ModuleList()
        self.fuse_net = cfg.get('fuse_net',None)
        self.src_sensor = cfg.get('src_sensor',None)
        self.src_sensor2 = cfg.get('src_sensor2',None)
        self.tgt_sensor = cfg.get('tgt_sensor',None)
        self.src_dir = cfg.get('src_dir',None)
        self.src2_dir = cfg.get('src2_dir',None)
        self.tgt_dir = cfg.get('tgt_dir',None)
        self.transformer_dir = cfg.get('transformer_dir',None)

        if self.ctx['z_type'] == 'discrete':
            self.ctx['z_tau_annealer'] = z_tau_annealer = ExpParamAnnealer(cfg.z_tau.start, cfg.z_tau.finish, cfg.z_tau.decay)
            self.param_annealers.append(z_tau_annealer)

        if not self.src2_dir:
            print('Using Sensor(s)',self.sensor)
            print('Fusionnet',self.fuse_net)
            print('Fusing Sensors',self.src_sensor,' to ',self.tgt_sensor)
            print('Loading transformer weights from',self.transformer_dir)
            print('Loading src_encoder weights from',self.src_dir)
            print('Loading tgt_encoder weights from',self.tgt_dir)
        else:
            print('Using Sensor(s)',self.sensor)
            print('Fusionnet',self.fuse_net)
            print('Fusing Sensors',self.src_sensor,self.src_sensor2,' to ',self.tgt_sensor)
            print('Loading transformer weights from',self.transformer_dir)
            print('Loading src_encoder weights from',self.src_dir)
            print('Loading src2_encoder weights from',self.src2_dir)
            print('Loading tgt_encoder weights from',self.tgt_dir)

        # save all computed variables
        self.data = None
        # map encoder
          
        if self.fuse_net:
            
            if self.fuse_net == 'transformer':
                
                self.radar_encoder = MapEncoder(cfg.map_encoder, ful_con = 196)
                self.lidar_encoder = MapEncoder(cfg.map_encoder, ful_con = 196)
                self.camera_encoder = MapEncoder(cfg.map_encoder, ful_con = 32)

                models = {'camera':self.camera_encoder,'lidar': self.lidar_encoder,'radar':self.radar_encoder}
                
                self.src_encoder = models[self.src_sensor]
                if self.src_sensor2: self.src_encoder2 = models[self.src_sensor2]
                self.tgt_encoder = models[self.tgt_sensor]


                # self.src_encoder = init_models(cfg, self.src_sensor)
                # if self.src_sensor2: self.src_encoder2 = init_models(cfg, self.src_sensor2)
                # self.tgt_encoder = init_models(cfg, self.tgt_sensor)
                # self.fusionnet = Transformer(32,dropout=0)

                self.src_encoder.load_state_dict(torch.load(self.src_dir))
                for param in self.src_encoder.parameters():
                    param.requires_grad = False

                if self.src_sensor2:
                    self.src_encoder2.load_state_dict(torch.load(self.src2_dir))
                    for param in self.src_encoder2.parameters():
                        param.requires_grad = False

                self.tgt_encoder.load_state_dict(torch.load(self.tgt_dir))
                for param in self.tgt_encoder.parameters():
                    param.requires_grad = False
                
                
                # self.fusionnet.load_state_dict(torch.load('/home/mide/msat/msat_transformer_2/transformer_camera_radar_large.pth'))
                # for param in self.fusionnet.parameters():
                #        param.requires_grad = False
                # first_layer_weights = next(self.fusionnet.transformer_1.parameters())
                # print(first_layer_weights.data)
                
                self.fusionnet = FuseModule(self.transformer_dir)
            
            elif self.fuse_net == 'fullyconnected':
                self.radar_encoder = MapEncoder(cfg.map_encoder, ful_con = 3600)
                self.lidar_encoder = MapEncoder(cfg.map_encoder, ful_con = 3600)
                self.camera_encoder = MapEncoder(cfg.map_encoder, ful_con = 720)
               
                self.fusionnet = FullNet()

            # self.ctx['map_enc_dim'] = self.tgt_encoder.out_dim
            self.ctx['map_enc_dim'] = 32
        else:
                self.ctx['map_enc_dim'] = self.radar_encoder.out_dim * 2 
            # increase this number as sensor size increases, effectcs FC layer in encoder/decoder networks (msat.py) -mg 


        # models
        self.context_encoder = ContextEncoder(cfg.context_encoder, self.ctx)
        self.future_encoder = FutureEncoder(cfg.future_encoder, self.ctx)
        self.future_decoder = FutureDecoder(cfg.future_decoder, self.ctx)
        
        

    def set_device(self, device):
        self.device = device
        self.to(device)

    def set_data(self, data):
        device = self.device 
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        if self.training and len(data['pre_motion_3D']) > self.max_train_agent:
            in_data = {}
            ind = np.random.choice(len(data['pre_motion_3D']), self.max_train_agent).tolist()
            for key in ['pre_motion_3D', 'fut_motion_3D', 'fut_motion_mask', 'pre_motion_mask', 'heading']:
                in_data[key] = [data[key][i] for i in ind if data[key] is not None]
        else:
            in_data = data

        self.data = defaultdict(lambda: None)
        self.data['batch_size'] = len(in_data['pre_motion_3D'])
        self.data['agent_num'] = len(in_data['pre_motion_3D'])
        self.data['pre_motion'] = torch.stack(in_data['pre_motion_3D'], dim=0).to(device).transpose(0, 1).contiguous()
        self.data['fut_motion'] = torch.stack(in_data['fut_motion_3D'], dim=0).to(device).transpose(0, 1).contiguous()
        self.data['fut_motion_orig'] = torch.stack(in_data['fut_motion_3D'], dim=0).to(device)   # future motion without transpose
        self.data['pre_motion_orig'] = torch.stack(in_data['pre_motion_3D'], dim=0).to(device) # past motion without transpose -mg
        #print ('pre_motion_tensor_bytes', self.data['pre_motion_orig'].element_size() * self.data['pre_motion_orig'].nelement()) #debug code
        self.data['fut_mask'] = torch.stack(in_data['fut_motion_mask'], dim=0).to(device)
        self.data['pre_mask'] = torch.stack(in_data['pre_motion_mask'], dim=0).to(device)
        scene_orig_all_past = self.cfg.get('scene_orig_all_past', False)
        if scene_orig_all_past:
            self.data['scene_orig'] = self.data['pre_motion'].view(-1, 2).mean(dim=0)
        else:
            self.data['scene_orig'] = self.data['pre_motion'][-1].mean(dim=0)
        if in_data['heading'] is not None:
            self.data['heading'] = torch.tensor(in_data['heading']).float().to(device)

        # rotate the scene
        if self.rand_rot_scene and self.training:
            if self.discrete_rot:
                theta = torch.randint(high=24, size=(1,)).to(device) * (np.pi / 12)
            else:
                theta = torch.rand(1).to(device) * np.pi * 2
            for key in ['pre_motion', 'fut_motion', 'fut_motion_orig']:
                self.data[f'{key}'], self.data[f'{key}_scene_norm'] = rotation_2d_torch(self.data[key], theta, self.data['scene_orig'])
            if in_data['heading'] is not None:
                self.data['heading'] += theta
        else:
            theta = torch.zeros(1).to(device)
            for key in ['pre_motion', 'fut_motion', 'fut_motion_orig']:
                self.data[f'{key}_scene_norm'] = self.data[key] - self.data['scene_orig']   # normalize per scene

        self.data['pre_vel'] = self.data['pre_motion'][1:] - self.data['pre_motion'][:-1, :]
        self.data['fut_vel'] = self.data['fut_motion'] - torch.cat([self.data['pre_motion'][[-1]], self.data['fut_motion'][:-1, :]])
        self.data['cur_motion'] = self.data['pre_motion'][[-1]]
        self.data['pre_motion_norm'] = self.data['pre_motion'][:-1] - self.data['cur_motion']   # normalize pos per agent
        self.data['fut_motion_norm'] = self.data['fut_motion'] - self.data['cur_motion']
        if in_data['heading'] is not None:
            self.data['heading_vec'] = torch.stack([torch.cos(self.data['heading']), torch.sin(self.data['heading'])], dim=-1)



        if self.use_map and self.sensor == 'all':
            if self.fuse_net == 'transformer':
               
                self.data['src']=torch.Tensor(data[self.src_sensor]).expand(1,-1,-1,-1).to(device) # gets src sensor data from data returned by dataloader. -mg
                if self.src_sensor2: self.data['src2']=torch.Tensor(data[self.src_sensor2]).expand(1,-1,-1,-1).to(device)
                self.data['tgt']=torch.Tensor(data[self.tgt_sensor]).expand(1,-1,-1,-1).to(device) # tgt and src data are images -mg 

                # adding batch to the image data -mg
                # batch is required for feature encoder input -mg
                # feeding masknet with 1 batch only so transformer can take 1,1,32 sequence -mg 
                
            else:
                self.data['lidar']=torch.Tensor(data['lidar'])
                self.data['radar']=torch.Tensor(data['radar']) 
                self.data['lidar']=self.data['lidar'].expand(self.data['batch_size'],-1,-1,-1).to(device) #expanding sensor tensor with batch size and rellocating in memory -mg
                self.data['radar']=self.data['radar'].expand(self.data['batch_size'],-1,-1,-1).to(device)#expanding sensor tensor with batch size and rellocating in memory -mg
                data['camera']=torch.Tensor(data['camera'])
                self.data['camera']=data['camera'].expand(self.data['batch_size'],-1,-1,-1).to(device).contiguous()  #expanding sensor tensor with batch size and rellocating in memory -mg

        elif self.use_map and self.sensor != 'all':
            data['sensors']=torch.Tensor(data['sensors']) 
            self.data['agent_sensors']=data['sensors'].expand(self.data['batch_size'],-1,-1,-1).to(device).contiguous()  #expanding sensor tensor with batch size and rellocating in memory -mg
            print('map_encoder input',self.data['agent_sensors'].shape()) #debug
            # takes sensor tensor we created earlier.
        # agent shuffling
        if self.training and self.ctx['agent_enc_shuffle']:
            self.data['agent_enc_shuffle'] = torch.randperm(self.ctx['max_agent_len'])[:self.data['agent_num']].to(device)
        else:
            self.data['agent_enc_shuffle'] = None

        conn_dist = self.cfg.get('conn_dist', 100000.0)
        cur_motion = self.data['cur_motion'][0]
        if conn_dist < 1000.0:
            threshold = conn_dist / self.cfg.traj_scale
            pdist = F.pdist(cur_motion)
            D = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]]).to(device)
            D[np.triu_indices(cur_motion.shape[0], 1)] = pdist
            D += D.T
            mask = torch.zeros_like(D)
            mask[D > threshold] = float('-inf')
        else:
            mask = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]]).to(device)
        self.data['agent_mask'] = mask

    def step_annealer(self):
        for anl in self.param_annealers:
            anl.step()

    def forward(self):
        if self.use_map and self.sensor == 'all':
          

            src_tensor =self.src_encoder(self.data['src']) # passing src and tgt data to feature encoders -mg
            if self.src_sensor2:
                src_tensor2 = self.src_encoder2(self.data['src2'])
            tgt_tensor = self.tgt_encoder(self.data['tgt']) # returns 1x32 tensor -mg

            if self.fuse_net == 'transformer':

                src_tensor = src_tensor.unsqueeze(dim=1) #Transformer need 3-dim tensor. -mg
                if self.src_sensor2:
                    src_tensor2 = src_tensor2.unsqueeze(dim=1)
                    src_tensor = torch.cat((src_tensor,src_tensor2),dim=0)
                tgt_tensor = tgt_tensor.unsqueeze(dim=1)
                fused_tensor = self.fusionnet(src_tensor,tgt_tensor)

                self.data['map_enc'] = fused_tensor.squeeze() # will return shape [32] -mg       
                self.data['map_enc']=self.data['map_enc'].expand(self.data['batch_size'],-1) # msat takes 2-dim tensor for map-encoder [B x 32]. -mg
    
                self.loss_tgt = fused_tensor.squeeze(dim = 0).detach().cpu() #predicted tensor to cpu -mg
                self.loss_gt =  tgt_tensor.squeeze(dim = 0 ).detach().cpu() #ground truth to cpu -mg
                distance = spatial.distance.cosine(self.loss_gt[0,:], self.loss_tgt[0,:]) #distance.cosine accepts 1-D array -mg
                self.acc = 1  - distance
            elif self.fuse_net == 'fullyconnected':
                merged_tensor = torch.cat((radar,lidar,camera),dim = -1)
                self.data['map_enc'] = self.fusionnet(merged_tensor)
                    
            elif self.fuse_net == None:
                self.data['map_enc'] = torch.cat((lidar,radar),dim=-1)
   
        elif self.use_map and self.sensor != 'all':
            print('map_encoder input',self.data['agent_sensors'].shape()) #debug
            self.data['map_enc']= self.map_encoder(self.data['agent_sensors']) 
        #sends sensor data to mapencoder -mg 
        # self.data['map_enc'] = self.map_encoder(self.data['agent_maps']) #feed map encoder with agent maps -mg
        self.context_encoder(self.data)
        self.future_encoder(self.data)
        self.future_decoder(self.data, mode='train', autoregress=self.ar_train)
        if self.compute_sample:         
            self.inference(sample_num=self.loss_cfg['sample']['k'])
        return self.data

    def inference(self, mode='infer', sample_num=20, need_weights=False):
        if self.use_map and self.data['map_enc'] is None:
            #self.data['map_enc'] = self.map_encoder(self.data['agent_sensors']) 
            
            src_tensor =self.src_encoder(self.data['src']) # passing src and tgt data to feature encoders -mg
            if self.src_sensor2:
                src_tensor2 = self.src_encoder2(self.data['src2'])
            tgt_tensor = self.tgt_encoder(self.data['tgt']) # returns 1x32 tensor -mg            

            src_tensor = src_tensor.unsqueeze(dim=1) #Transformer need 3-dim tensor. -mg
            if self.src_sensor2:
                src_tensor2 = src_tensor2.unsqueeze(dim=1)
                src_tensor = torch.cat((src_tensor,src_tensor2),dim=0)
            tgt_tensor = tgt_tensor.unsqueeze(dim=1)
            fused_tensor = self.fusionnet(src_tensor,tgt_tensor)

            self.data['map_enc'] = fused_tensor.squeeze() # will return shape [32] -mg
            
            self.data['map_enc']=self.data['map_enc'].expand(self.data['batch_size'],-1) # msat takes 2-dim tensor for map-encoder [B x 32]. -mg

            self.loss_tgt = fused_tensor.squeeze(dim = 0).detach().cpu() #predicted tensor to cpu -mg
            self.loss_gt =  tgt_tensor.squeeze(dim = 0 ).detach().cpu() #ground truth to cpu -mg
            distance = spatial.distance.cosine(self.loss_gt[0,:], self.loss_tgt[0,:]) #distance.cosine accepts 1-D array -mg
            self.acc = 1  - distance
            print('acc', self.acc)
            
                     
        if self.data['context_enc'] is None:
            self.context_encoder(self.data)
        if mode == 'recon':
            sample_num = 1
            self.future_encoder(self.data)
        self.future_decoder(self.data, mode=mode, sample_num=sample_num, autoregress=True, need_weights=need_weights)
        return self.data[f'{mode}_dec_motion'], self.data

    def compute_loss(self):
        total_loss = 0
        #loss_dict = {} #causes memory leak -mg
        loss_unweighted_dict = {}
        if self.fuse_net == 'transformer': transformer_acc = self.acc 
        else: transformer_acc = 0
        for loss_name in self.loss_names:
            loss, loss_unweighted = loss_func[loss_name](self.data, self.loss_cfg[loss_name])
            total_loss += loss
            #loss_dict[loss_name] = loss.item()
            loss_unweighted_dict[loss_name] = loss_unweighted.item()
        return total_loss, loss_unweighted_dict,transformer_acc

    #TODO: make transformer acc logg avalaible only on transformer mode
    # delete unrelated comments


