import torch
torch.manual_seed(0)
from torch import nn
from model.fuse_network import Transformer
from model.context_learner import ContextNet
from model.map_encoder import MapEncoder
from utils.config import Config
from sklearn.metrics import accuracy_score
import os 
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
#from Levenshtein import distance
from scipy import spatial
from model.sequence_folders import SequenceFolder
from model.sequence_folders_lidar import SequenceFolder_lidar
from model.sequence_folders_mono import SequenceFolder_mono
from torch import optim

class EarlyStopper:
    def __init__(self, patience=400, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss,model,exp_name):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(model.state_dict(),'/home/mide/msat/msat_transformer/transformer_{}.pth'.format(exp_name))
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
           
        return self.counter

#TODO: 
# add sensor options to cfg
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src1', default=None)
    parser.add_argument('--src2', default='False')
    parser.add_argument('--tgt', default=None)
    parser.add_argument('--cfg', default=None)
    parser.add_argument('--mode', default='train')
    parser.add_argument('--output_dim', type=int, default=32)
    parser.add_argument('--exp_name', default='none')
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--workers',type = int,default = 4)
    parser.add_argument('--normalize', default=True)
    parser.add_argument('--root',default='/media/nfs/transformer_train')
    args = parser.parse_args()

    print('Initializing Code...')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    output_dim = args.output_dim #output_dim
    cfg = Config(args.cfg, create_dirs=True)

    # cfg = output_dim
    normalize = args.normalize
    mode = args.mode
    data_root = args.root
    exp_name = args.exp_name
    src = args.src1
    tgt = args .tgt
    src2 = args.src2

    ro_params = {
            'cart_resolution': 0.25,
            'cart_pixels': 512,
            'rangeResolutionsInMeter':  0.175,
            'angleResolutionInRad': 0.015708,
            'radar_format': 'polar'
        }
    

    '''
    Init dataloaders
    '''
    print('Initializing Training data loaders')
    sensor_radar = SequenceFolder(
        os.path.join(data_root,'sequences/'),
        transform=None, 
        seed=0,
        mode='train',
        sequence_length=3,
        skip_frames=1,
        dataset='robotcar',
        ro_params=ro_params,
        load_camera=None,    
        cam_mode= 'mono',
        cam_transform=None, 
        cam_preprocessed=1
        )

    sensor_lidar = SequenceFolder_lidar(os.path.join(data_root,'sequences/'),
    transform=None, 
    seed=0,
    lo_params = {
    'cart_pixels': 512},
    mode='train',
    sequence_length=3,
    skip_frames=1,
    dataset='robotcar',
    load_camera=False,    
    cam_mode= 'mono',
    cam_transform=None, 
    cam_preprocessed=False)
            
    sensor_camera= SequenceFolder_mono(os.path.join(data_root,'sequences/'),dataset='robotcar', 
    seed=None, train=True,
    sequence_length=3, transform=None, skip_frames=1, preprocessed=True)

    radar_loader = torch.utils.data.DataLoader(
        sensor_radar, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=None)
    
   

    lidar_loader = torch.utils.data.DataLoader(
        sensor_lidar, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=None)
    
    camera_loader = torch.utils.data.DataLoader(
        sensor_camera, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=None)
    
    loaders = {'camera': camera_loader,'radar':radar_loader,'lidar':lidar_loader}

    #init validation loss 

    print('Initializing Validation data loaders')
    val_radar = SequenceFolder(
        os.path.join(data_root,'sequences/'),
        transform=None, 
        seed=0,
        mode='validation',
        sequence_length=3,
        skip_frames=1,
        dataset='robotcar',
        ro_params=ro_params,
        load_camera=None,    
        cam_mode= 'mono',
        cam_transform=None, 
        cam_preprocessed=1
        )

    val_lidar = SequenceFolder_lidar(os.path.join(data_root,'sequences/'),
    transform=None, 
    seed=0,
    lo_params = {
    'cart_pixels': 512},
    mode='validation',
    sequence_length=3,
    skip_frames=1,
    dataset='robotcar',
    load_camera=False,    
    cam_mode= 'mono',
    cam_transform=None, 
    cam_preprocessed=False)
            
    val_camera= SequenceFolder_mono(os.path.join(data_root,'sequences/'),dataset='robotcar', 
    seed=None, train=False,
    sequence_length=3, transform=None, skip_frames=1, preprocessed=True)

    radar_val = torch.utils.data.DataLoader(
        val_radar, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=None)

    lidar_val = torch.utils.data.DataLoader(
        val_lidar, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=None)
    camera_val = torch.utils.data.DataLoader(
        val_camera, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=None)

    val_loaders = {'camera': camera_val,'radar':radar_val,'lidar':lidar_val}

    '''
    Init Models
    '''
    print('Initializing models')

    if src == 'camera': 
        ful_con_src = 6688
        ful_con_tgt = 30752
    elif tgt ==  'camera':
        ful_con_src = 30752
        ful_con_tgt = 6688
    else:
        ful_con_src = 30752 
        ful_con_tgt = 30752
    
    if src2 != 'False':
 

        if src2 == 'camera': ful_con_src2 = 6688
        else: ful_con_src2 = 30752
        src_encoder2 = MapEncoder(cfg.map_encoder, ful_con = ful_con_src2) #setup encoder 2 for camera src if required -mg
        src_encoder2.to(device)
        loader_mode = 'multi'

    else:
        src_loader = loaders[src]  
        loader_mode = 'single'


    src_encoder = MapEncoder(cfg.map_encoder, ful_con = ful_con_src)
    tgt_encoder = MapEncoder(cfg.map_encoder, ful_con = ful_con_tgt)

    src_encoder.to(device)
    tgt_encoder.to(device)


    transformer = Transformer(32,dropout = 0.4)
    transformer.to(device)
    stop = EarlyStopper()
    save_dir = '/home/mide/msat/msat_transformer'
    tb_logger = SummaryWriter('{}/logs/{}'.format(save_dir,exp_name))
    results = []
  

    '''
    Define training function.
    '''
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(transformer.parameters(), lr=1.e-3)

    acc = 0
    total_loss = 0
    total_rmse = 0

    transformer.train()

    for epoch in range(args.epochs):

        print('Starting Epoch {}'.format(epoch))
        if loader_mode == 'single':
            zips = zip(loaders[src],loaders[tgt])

        else:
            zips = zip(loaders[src],loaders[src2],loaders[tgt])
          
        i = 0
        tensors = []
        for i, (tensors) in enumerate (zips):

            #with torch.autograd.detect_anomaly():
                if len(tensors) == 3:
                    src_tensor_1 = tensors[0] #scr_loader_1 
                    src_tensor_2 = tensors[1] #src_loader_2
                    tgt_tensor = tensors[2] # tgt_loader
                    src_tensor_1  = src_tensor_1.to(device)
                    src_tensor_2  = src_tensor_2.to(device)
                    tgt_tensor = tgt_tensor.to(device)
                else: 
                    src_tensor = tensors[0]
                    tgt_tensor = tensors[1]
                    src_tensor  = src_tensor.to(device)
                    tgt_tensor = tgt_tensor.to(device)

                if  stop.counter != stop.patience:
                
                    if len(tensors) == 3:
                        
                        f_src_1 = src_encoder(src_tensor_1).unsqueeze(dim = 1)  #add batch, to mid dimension
                        f_src_2 = src_encoder2(src_tensor_2).unsqueeze(dim = 1)
                        f_src = torch.cat((f_src_1,f_src_2),dim=0) #squeeze them with dim=0, increases sequence length
                    
                        

                    else:
                        f_src = src_encoder(src_tensor).unsqueeze(dim = 1)

                    
                    f_tgt = tgt_encoder(tgt_tensor).unsqueeze(dim = 1)       
                    fused = transformer(f_src,f_tgt).squeeze(dim = 1)   #takes src and tgt tensor in order -mg  
        
                    f_tgt = f_tgt.squeeze(dim = 1) # squeeze tgt tensor
                    loss = loss_fn(fused,f_tgt) #calculate loss between prediction and tgt tensor -mg
                    rmse = torch.sqrt(loss)

                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    #calculate accuracy
                    reference = f_tgt.tolist() #take tgt tensor as reference -mg
                    candidate = fused.tolist()
                    distance = spatial.distance.cosine(reference[0], candidate[0])
                    #train_acc = 1 - distance / max(len(reference), len(candidate))
                    train_acc = 1 - distance
                    acc += train_acc
                    total_loss += loss
                    total_rmse +=rmse
                    if i%100 == 0:
                        # print(fused.shape,f_tgt.shape)
                        print('src',torch.isnan(f_src).any(),'transformer',torch.isnan(fused).any(),'tgt',torch.isnan(f_tgt).any())
                        # breakpoint()
                        
                        stop.counter = stop.early_stop(total_loss/100,transformer,exp_name)
                        
                        print('Epoch {} Iteration {}/{} Train_Loss {} RMSE {} Train_acc {}'.format(epoch,i,len(loaders[tgt]),total_loss/100,total_rmse/100,acc/100))
                        tb_logger.add_scalar('learning loss', total_loss/100,i) 
                        tb_logger.add_scalar('rmse',total_rmse/100,i)
                        tb_logger.add_scalar('train accuracy',acc/100,i)
                        acc = 0
                        total_loss = 0
                        total_rmse = 0 
                        
                else:  
                    print('applied early stop')
                    
                    torch.save(tgt_encoder.state_dict(),'{}/{}_tgt_encoder_{}.pth'.format(save_dir,tgt,exp_name))
                    torch.save(src_encoder.state_dict(),'{}/{}_src_encoder_{}.pth'.format(save_dir,src,exp_name))
                    if loader_mode == 'multi':torch.save(src_encoder2.state_dict(),'{}/{}_src_encoder2_{}.pth'.format(src2,exp_name))
                    break
        if stop.counter == stop.patience: break
        print('End of epoch {}'.format(epoch))
        print('Saving models')
        torch.save(tgt_encoder.state_dict(),'{}/{}_tgt_encoder_{}.pth'.format(save_dir,tgt,exp_name))
        torch.save(src_encoder.state_dict(),'{}/{}_src_encoder_{}.pth'.format(save_dir,src,exp_name))
        if loader_mode == 'multi':torch.save(src_encoder2.state_dict(),'{}/{}_src_encoder2_{}.pth'.format(save_dir,src2,exp_name))
        torch.save(transformer.state_dict(),'{}/transformer_{}.pth'.format(save_dir
                                                                     ,exp_name))

            
                

