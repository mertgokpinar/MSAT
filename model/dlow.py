import torch
from torch import nn
from torch.nn import functional as F
from utils.torch import *
from utils.config import Config
from .common.mlp import MLP
from .common.dist import *
from . import model_lib
from .fuse_network import Transformer
from .fuse_sensor import FuseModule
from .FullNet import FullNet
from scipy import spatial
import os
from .map_encoder import MapEncoder
#torch.manual_seed(0)



def compute_z_kld(data, cfg):
    loss_unweighted = data['q_z_dist_dlow'].kl(data['p_z_dist_infer']).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= data['batch_size']
    loss_unweighted = loss_unweighted.clamp_min_(cfg.min_clip)
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def diversity_loss(data, cfg):
    loss_unweighted = 0
    fut_motions = data['infer_dec_motion'].view(*data['infer_dec_motion'].shape[:2], -1)
    for motion in fut_motions:
        dist = F.pdist(motion, 2) ** 2
        loss_unweighted += (-dist / cfg['d_scale']).exp().mean()
    if cfg.get('normalize', True):
        loss_unweighted /= data['batch_size']
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def recon_loss(data, cfg):

    diff = data['infer_dec_motion'] - data['fut_motion_orig'].unsqueeze(1)
    if cfg.get('mask', True):
        mask = data['fut_mask'].unsqueeze(1).unsqueeze(-1)
        diff *= mask
    dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
    loss_unweighted = dist.min(dim=1)[0]
    if cfg.get('normalize', True):
        loss_unweighted = loss_unweighted.mean()
    else:
        loss_unweighted = loss_unweighted.sum()
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


loss_func = {
    'kld': compute_z_kld,
    'diverse': diversity_loss,
    'recon': recon_loss,
}


""" DLow (Diversifying Latent Flows)"""
class DLow(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device('cpu')
        self.cfg = cfg
        self.nk = nk = cfg.sample_k
        self.nz = nz = cfg.nz
        self.share_eps = cfg.get('share_eps', True)
        self.train_w_mean = cfg.get('train_w_mean', False)
        self.loss_cfg = self.cfg.loss_cfg
        self.loss_names = list(self.loss_cfg.keys())
        self.fuse_net = cfg.get('fuse_net',None)
        self.src_sensor = cfg.get('src_sensor',None)
        self.tgt_sensor = cfg.get('tgt_sensor',None)
        self.src_sensor2 = cfg.get('src_sensor2',None)
        self.src_dir = cfg.get('src_dir',None)
        self.tgt_dir = cfg.get('tgt_dir',None)
        self.transformer_dir = cfg.get('transformer_dir',None)

        pred_cfg = Config(cfg.pred_cfg, tmp=False, create_dirs=False)
        pred_model = model_lib.model_dict[pred_cfg.model_id](pred_cfg)
        self.pred_model_dim = pred_cfg.tf_model_dim
        if cfg.pred_epoch > 0:
            cp_path = pred_cfg.model_path % cfg.pred_epoch
            print('loading model from checkpoint: %s' % cp_path)
            model_cp = torch.load(cp_path, map_location='cpu')
            pred_model.load_state_dict(model_cp['model_dict'])

            pred_model.eval()
        #   pred_model.fusionnet.eval()
        # first_layer_weights = next(pred_model.fusionnet.transformer_1.parameters())
        # print(first_layer_weights.data)
        # breakpoint()
   
        self.pred_model = [pred_model]
        # Dlow's Q net
        self.qnet_mlp = cfg.get('qnet_mlp', [512, 256])
        self.q_mlp = MLP(self.pred_model_dim, self.qnet_mlp)
        self.q_A = nn.Linear(self.q_mlp.out_dim, nk * nz)
        self.q_b = nn.Linear(self.q_mlp.out_dim, nk * nz)
        
    def set_device(self, device):
        self.device = device
        self.to(device)
        self.pred_model[0].set_device(device)

    def set_data(self, data):
        self.pred_model[0].set_data(data)
        self.data = self.pred_model[0].data
        
  
    def main(self, mean=False, need_weights=False):
        pred_model = self.pred_model[0]
       
        if hasattr(pred_model, 'use_map') and pred_model.use_map and pred_model.sensor == 'all' and self.fuse_net == 'transformer':
         
                    

            src_tensor = pred_model.src_encoder(self.data['src']) # passing src and tgt data to feature encoders -mg
            if self.src_sensor2: src_tensor2 = pred_model.src_encoder2(self.data['src2'])
            tgt_tensor = pred_model.tgt_encoder(self.data['tgt']) # returns 1x32 tensor -mg
            
            
            src_tensor = src_tensor.unsqueeze(dim= 0) #Transformer need 3-dim tensor. -mg
            tgt_tensor = tgt_tensor.unsqueeze(dim =0 )
            
            
            if self.src_sensor2: src_tensor2 = src_tensor2.unsqueeze(dim = 0)
            if self.src_sensor2: src_tensor = torch.cat((src_tensor,src_tensor2),dim=0)  
            #fused_tensor = self.fusionnet(src_tensor, tgt_tensor)
                        
            fused_tensor = pred_model.fusionnet(src_tensor, tgt_tensor)
            self.data['map_enc'] = fused_tensor.squeeze() # will return shape [32]
            self.data['map_enc']=self.data['map_enc'].expand(self.data['batch_size'],-1)
            self.loss_tgt = fused_tensor.squeeze(dim = 0).detach().cpu() 
            self.loss_gt =  tgt_tensor.squeeze(dim = 0).detach().cpu() #ground truth to cpu -mg
            distance = spatial.distance.cosine(self.loss_gt[0,:], self.loss_tgt[0,:])
            self.acc = 1  - distance
            
            # print('tgt tensor', tgt_tensor)
            # print('src tensor', src_tensor)
            # print('fused_sensor',fused_tensor)
            # print('acc', self.acc)
            # breakpoint()


            
        if hasattr(pred_model, 'use_map') and pred_model.use_map and pred_model.sensor == 'all' and self.fuse_net == 'fullyconnected':
           
            radar = pred_model.radar_encoder(self.data['radar'])
            lidar = pred_model.lidar_encoder(self.data['lidar'])
            camera = pred_model.camera_encoder(self.data['camera'])
            tensor = torch.cat((radar,lidar,camera),dim = -1)
            self.data['map_enc'] = pred_model.fusionnet(tensor)
        elif hasattr(pred_model, 'use_map') and pred_model.use_map and pred_model.sensor == 'all' and self.fuse_net == 'None':
            
            self.data['map_enc'] = torch.cat((lidar,radar),dim=-1)

        elif self.fuse_net == 'None' and pred_model.use_map:
           
            self.data['map_enc'] = pred_model.map_encoder(self.data['agent_sensors'])

        pred_model.context_encoder(self.data)

        if not mean:
            if self.share_eps:
                eps = torch.randn([1, self.nz]).to(self.device)
                eps = eps.repeat((self.data['agent_num'] * self.nk, 1))
            else:
                eps = torch.randn([self.data['agent_num'], self.nz]).to(self.device)
                eps = eps.repeat_interleave(self.nk, dim=0)

        qnet_h = self.q_mlp(self.data['agent_context'])
        A = self.q_A(qnet_h).view(-1, self.nz)
        b = self.q_b(qnet_h).view(-1, self.nz)

        z = b if mean else A*eps + b
        logvar = (A ** 2 + 1e-8).log()
        self.data['q_z_dist_dlow'] = Normal(mu=b, logvar=logvar)

        pred_model.future_decoder(self.data, mode='infer', sample_num=self.nk, autoregress=True, z=z, need_weights=need_weights)
        return self.data
    
    def forward(self):
        return self.main(mean=self.train_w_mean)

    def inference(self, mode, sample_num, need_weights=False):
        self.main(mean=True, need_weights=need_weights)
        res = self.data[f'infer_dec_motion']
        if mode == 'recon':
            res = res[:, 0]
        return res, self.data

    def compute_loss(self):
        total_loss = 0
        # loss_dict = {}
        loss_unweighted_dict = {}
        if self.fuse_net == 'transformer': transformer_acc = self.acc 
        else: transformer_acc = 0
        for loss_name in self.loss_names:
            loss, loss_unweighted = loss_func[loss_name](self.data, self.loss_cfg[loss_name])
            total_loss += loss
            # loss_dict[loss_name] = loss.item()
            loss_unweighted_dict[loss_name] = loss_unweighted.item()
        return total_loss, loss_unweighted_dict,transformer_acc

    def step_annealer(self):
        pass