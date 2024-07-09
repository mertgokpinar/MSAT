import argparse
import os
import sys
import time
import matplotlib.pyplot as plt
import sklearn.metrics
from matplotlib import cm 
import numpy as np
import torch
from sklearn.metrics import pairwise_distances_argmin_min
import os
import random
#torch.manual_seed(0)
from torch import get_num_interop_threads, optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import yaml
sys.path.append(os.getcwd())
from data.dataloader import data_generator
from model.model_lib import model_dict
from utils.config import Config
from utils.torch import *
from utils.utils import (AverageMeter, convert_secs2time, get_timestring,
                         prepare_seed, print_log)

from model.msat_single import msat_single
from model.dlow import DLow
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
from test import test_model
import subprocess

#TODO: move early_stop and tensorboard defs to utils.

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#setup_seed(0)

class EarlyStopper:
    def __init__(self, patience=10, penalty=0.04):
        self.patience = patience
        self.penalty = penalty
        self.counter = 0
        self.min_train_loss = np.inf

    def stop(self, train_loss):
        if train_loss + self.penalty < self.min_train_loss:
            self.min_train_loss = train_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True,self.counter
           
        return None,self.counter

def logging(cfg, epoch, total_epoch, iter, total_iter, ep, seq, frame, losses_str, log,transformer_acc,patient):
	print_log('{} | Epo: {:02d}/{:02d}, '
		'It: {:04d}/{:04d}, '
		'EP: {:s}, ETA: {:s}, seq {:s}, frame {:05d}, {}, transformer_acc: {}, patient_count {}'
        .format(cfg, epoch, total_epoch, iter, total_iter, \
		convert_secs2time(ep), convert_secs2time(ep / iter * (total_iter * (total_epoch - epoch) - iter)), seq, frame, losses_str,transformer_acc, patient), log) #add output_t after losses_str -mg

def feature_maps(processed):
    
    processed = processed.detach().cpu()

    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i+1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
    return fig 


def traj2Fig_withgt(pred_xy, gt_xy,past_xy, axes=[0, 1]): #use this def to plot trajectory with gt data -mg
    """Make `matplotlib.pyplot.figure` from the 2D plot of a given trajectory.

    Args:
        pred_xy (torch.Tensor): Trajectory to plot. Shape: [N,3]
        gt_xy (torch.Tensor): Trajectory to plot. Shape: [N,3]

    Returns:
        matplotlib.pyplot.figure: Figure of the trajectory plot
    """
    
    fig, axs = plt.subplots(2,1)
    
    axs[0].scatter(pred_xy[:, 0], pred_xy[:, 1])
    axs[0].set_title('Predicted Traj scatter')
    axs[0].set_xscale("linear") 
    axs[0].set_yscale("linear") 
    axs[0].ticklabel_format(useOffset=False)
    axs[1].scatter(gt_xy[:, 0], gt_xy[:, 1],label='GT')
    axs[1].scatter(past_xy[:, 0], past_xy[:, 1],label='past')
    axs[1].set_title('GT future and past traj. scatter')
    axs[1].legend(fontsize=6)
    axs[1].set_xscale("linear") 
    axs[1].set_yscale("linear") 
    axs[1].ticklabel_format(useOffset=False)
    plt.tight_layout()

    return fig

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

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)

        array = tensor.detach().cpu().numpy()
    return array

def MultiModal_loss(model1_predictions, model2_predictions, labels):
    model1_predictions = model1_predictions[0].squeeze(dim = 0)
    model2_predictions = model2_predictions[0].squeeze(dim = 0)
    labels =labels[0].squeeze(dim=0)
    breakpoint() 
    loss1 = nn.BCELoss()(model1_predictions, labels)
    loss2 = nn.BCELoss()(model2_predictions, labels.long()) 
    total_loss = (1 * loss1) + (1 * loss2)   
    return total_loss

def Pairwise_loss(base_model_predictions, new_model_predictions, margin=1.0):
    pairwise_diff = new_model_predictions - base_model_predictions    
    pairwise_distances = torch.norm(pairwise_diff, p=2, dim=1)  
    pairwise_loss = torch.clamp(margin - pairwise_distances, min=0)   
    mean_loss = torch.mean(pairwise_loss)
    
    return mean_loss


def train(epoch,patient):
    global tb_ind
    since_train = time.time()
    generator.shuffle()
    train_loss_meter = {x: AverageMeter() for x in cfg.loss_cfg.keys()}
    train_loss_meter['total_loss'] = AverageMeter()
    last_generator_index = 0
    avg_loss = 0
    print('hellooooo')
    while not generator.is_epoch_end():
        data = generator() #generator returns current index's data object. -mg
        # training reads code in this lines and returns code, data{} comes from preprocessor.py -mg
        if data is not None:
            seq, frame = data['seq'], data['frame']
            model.set_data(data) #msat.py def set_data() -mg
            model_data = model() 
            
            if model_id=='dlow':
                fut_predic=model_data['infer_dec_motion'][0,0].detach().cpu()
                fut_gt=model_data['fut_motion_orig'][0,:,:].detach().cpu()
                past_xy=model_data['pre_motion_orig'][0,:,:].detach().cpu()                     
            total_loss,loss_unweighted_dict,transformer_acc = model.compute_loss() #def in msat.py -mg
            total_loss = total_loss
            optimizer.zero_grad()
            total_loss.backward() 
            optimizer.step()
            train_loss_meter['total_loss'].update(total_loss.item())
                     
            for key in loss_unweighted_dict.keys():
                train_loss_meter[key].update(loss_unweighted_dict[key])
          
        if generator.index - last_generator_index > cfg.print_freq: 
            ep = time.time() - since_train
            losses_str = ' '.join([f'{x}: {y.avg:.3f} ({y.val:.3f})' for x, y in train_loss_meter.items()])
            logging(args.cfg, epoch, cfg.num_epochs, generator.index, generator.num_total_samples, ep, seq, frame, losses_str, log, transformer_acc,patient )

            #tb_logger.add_scalar('memmory usage',torch.cuda.memory_allocated(),tb_ind)
            tb_logger.add_scalar('transfrormer accuracy', transformer_acc,tb_ind)
            
            if model_id=='dlow':
                fig = traj2Fig_withgt(fut_predic.squeeze(),fut_gt.squeeze(),past_xy.squeeze())
                tb_logger.add_figure('predicted_fut_traj',fig,tb_ind) #summary writer trajectory figure -mg
                tb_logger.add_histogram('fut_gt_traj_hist_(x-axis)',fut_gt[:,0],tb_ind) #summary writer future gt&predicted histogram -mg
                tb_logger.add_histogram('fut_pred_traj_hist_(x-axis)',fut_predic[:,0],tb_ind)
                tb_logger.add_histogram('fut_gt_traj_hist_(y-axis)',fut_gt[:,1],tb_ind) 
                tb_logger.add_histogram('fut_pred_traj_hist_(y-axis)',fut_predic[:,1],tb_ind)            
                if cfg.get('load_map') and cfg.get('fuse_net')==False:
                    tb_logger.add_image('masknet_output',tensor2array(model_data['map_enc']),tb_ind)
                    tb_logger.add_image('masknet_input',tensor2array(model_data['agent_sensors'][0]),tb_ind) #map encoder input radar image -mg
            
            for name, meter in train_loss_meter.items():
                tb_logger.add_scalar('model_' + name, meter.avg, tb_ind)
            tb_ind += 1
            last_generator_index = generator.index

    avg_loss= train_loss_meter['total_loss'].avg
    scheduler.step()
    model.step_annealer()
    tb_logger.add_scalar('Epoch avarage loss',avg_loss,epoch)
    tb_logger.add_scalar('memmory usage',torch.cuda.memory_allocated(),epoch)
    tb_logger.add_scalar('Samples per epoch',train_loss_meter['total_loss'].count,epoch) #remove later
    return avg_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg, args.tmp, create_dirs=True)
    #prepare_seed(cfg.seed)parser.add_argument('--mode', default='train')
    setup_seed(0)
    torch.set_default_dtype(torch.float32)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    time_str = get_timestring()
    log = open(os.path.join(cfg.log_dir, 'log.txt'), 'a+')
    print_log("time str: {}".format(time_str), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)
    tb_logger = SummaryWriter(cfg.tb_dir) #defines summarywriter as log directory -mg
    tb_ind = 0

    """ data """
    generator = data_generator(cfg, log, split='train', phase='training')
    
    """ model """
    model_id = cfg.get('model_id')
    fuse_network = cfg.get('fuse_net','transformer')
    model = model_dict[model_id](cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr) #we are using adam as optimizer
    scheduler_type = cfg.get('lr_scheduler', 'linear')
    
    if scheduler_type == 'linear':
       scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.lr_fix_epochs, nepoch=cfg.num_epochs)
    elif scheduler_type == 'step':
        scheduler = get_scheduler(optimizer, policy='step', decay_step=cfg.decay_step, decay_gamma=cfg.decay_gamma)
    else:
        raise ValueError('unknown scheduler type!')

    # #load pre-trained encoder and decoders -mg
         
    # if cfg.optimization != False:    
    #     pre_path = '/home/mide/msat/results/MSAT_RESULTS/msat-nosensor-exp4/msat-nosensor-base/models/model_0030.p'
    #     model_cp = torch.load(pre_path, map_location='cpu')
    #     print('LOADED ENCODERS')
    #     model.context_encoder.load_state_dict(model_cp['context_encoder_dict'])
    #     for param in model.context_encoder.parameters():
    #         param.requires_grad = False
    #     model.future_encoder.load_state_dict(model_cp['future_encoder_dict'])
    #     for param in model.future_encoder.parameters():
    #         param.requires_grad = False
    #     model.future_decoder.load_state_dict(model_cp['future_decoder_dict'])
    #     for param in model.future_decoder.parameters():
    #         param.requires_grad = False
            
    
    model.set_device(device)
    if args.start_epoch > 0:
        cp_path = cfg.model_path % args.start_epoch
        print_log(f'loading model from checkpoint: {cp_path}', log)
        model_cp = torch.load(cp_path, map_location='cpu')
        model.load_state_dict(model_cp['model_dict'])
        if 'opt_dict' in model_cp:
            optimizer.load_state_dict(model_cp['opt_dict'])
            # for state in optimizer.state.values():
            #     for k, v in state.items():
            #         if isinstance(v, torch.Tensor):
            #             state[k] = v.cuda()
        if 'scheduler_dict' in model_cp:
            scheduler.load_state_dict(model_cp['scheduler_dict'])
    """ start training """ 
    print('/////MODEL SUMMARY/////')
    print(model_id)
    print(summary(model))
    model.train()
    patient = 0 
    for i in range(args.start_epoch, cfg.num_epochs):
        ep_loss = train(i,patient)   
        """ save model without earlystop"""
        if cfg.model_save_freq > 0 and (i + 1) % cfg.model_save_freq == 0:
            cp_path = cfg.model_path % (i + 1)
            if  model_id == 'dlow':
                model_cp = {'model_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(), 
                'scheduler_dict': scheduler.state_dict(),
                'epoch': i + 1}
            else:
                model_cp = {'model_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(), 
                'scheduler_dict': scheduler.state_dict(), 'context_encoder_dict':model.context_encoder.state_dict(),
                'future_encoder_dict':model.future_encoder.state_dict(),
                'future_decoder_dict':model.future_decoder.state_dict(),
                'epoch': i + 1}
            torch.save(model_cp,cp_path)
            #cmd = f"python test.py --cfg {args.cfg} --data_eval val --gpu 0"
            #subprocess.run(cmd.split(' '))
            
