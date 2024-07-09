import torch 
torch.manual_seed(0) #best performing seed was 200
from torch import nn
from model.fuse_network import Transformer
from model.context_learner import ContextNet
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
from model.map_encoder import MapEncoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src1', default=None)
    parser.add_argument('--src2', default='False')
    parser.add_argument('--tgt', default=None)
    parser.add_argument('--src1_dir', default='') #change model paths !
    parser.add_argument('--src2_dir', default='')
    parser.add_argument('--tgt_dir', default='')
    parser.add_argument('--output_dim', type=int, default=32)
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--workers',type = int,default = 4)
    parser.add_argument('--normalize', default=True)
    parser.add_argument('--root',default='/media/otonom/cold_storage/radiate')
    args = parser.parse_args()

    print('Initializing Code...')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    output_dim = args.output_dim #output_dim
    cfg = output_dim
    normalize = args.normalize
    data_root = args.root
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
        mode='test',
        sequence_length=3,
        skip_frames=1,
        dataset='radiate',
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
    mode='test',
    sequence_length=3,
    skip_frames=1,
    dataset='radiate',
    load_camera=False,    
    cam_mode= 'mono',
    cam_transform=None, 
    cam_preprocessed=False)
            
    sensor_camera= SequenceFolder_mono(os.path.join(data_root,'sequences/'),dataset='radiate', 
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
    '''
    Init Models
    '''
    print('Initializing models')
    radar_extractor = ContextNet(cfg, num_channels=1, ful_con=3600) # original was ContextNet and add num_channels=1 argument -mg
    lidar_extractor = ContextNet(cfg, num_channels=1, ful_con=3600)
    #camera_extractor = ContextNet(cfg, num_channels=1, ful_con=720)
    radar_extractor.to(device)
    lidar_extractor.to(device)
    dropout = nn.Dropout(0.5) #droupt value for feature extractor ! -mg
    out_dim = output_dim
    input_size = 32
    transformer = Transformer(32,dropout=0)
    transformer.load_state_dict(torch.load(model_dir))
    for param in transformer.parameters():
        param.requires_grad = False
    transformer.to(device)

    if src == 'camera': 
        ful_con_src = 720 
        ful_con_tgt = 3600
    elif tgt ==  'camera':
        ful_con_src = 3600 
        ful_con_tgt = 720
    else:
        ful_con_src = 3600 
        ful_con_tgt = 3600
 


    src_encoder = MapEncoder(cfg.map_encoder, ful_con = ful_con_src)
    tgt_encoder = MapEncoder(cfg.map_encoder, ful_con = ful_con_tgt)
    src_encoder.load_state_dict(torch.load(src_dir))
    for param in src_encoder.parameters():
        param.requires_grad = False

    tgt_encoder.load_state_dict(torch.load(tgt_dir))

    for param in tgt_encoder.parameters():
        param.requires_grad = False   

    src_encoder.to(device)
    tgt_encoder.to(device)

    results = []
    
    tb_logger = SummaryWriter('/home/otonom/msat/transformer_model/test-logs/radiate_dataset')

    if src2 != 'False':

        src_loader_1 = loaders[src]
        src_loader_2 = loaders[src2]
        if src or src2 == 'camera': ful_con_src2 = 720 
        else: ful_con_src2 = 3600
        src_encoder2 = MapEncoder(cfg.map_encoder, ful_con = ful_con_src2) #setup encoder 2 for camera src if required -mg
        src_encoder2.to(device)
        src_encoder2.load_state_dict(torch.load(src2_dir))
        for param in src_encoder.parameters():
            param.requires_grad = False
        loader_mode = 'multi'

    else:
        src_loader = loaders[src]  
        loader_mode = 'single'

    tgt_loader = loaders[tgt]

    with torch.no_grad():
        transformer.eval()
        if loader_mode == 'single':
            zip = zip(src_loader,tgt_loader)
            tensors = []
        else:
            zip = zip(src_loader_1,src_loader_2,tgt_loader)
            tensors = []

        for epoch in range(args.epochs):
        
            transformer.train()
            for i, (tensors) in enumerate (zip):
                
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


               
                if len(tensors) == 3:
                    if ful_con_src2 == 720:
                        if src == 'camera': 
                            encoder2 = src_tensor_1 #encoder2 is defined for camera, src_tensor_1 comes from src
                            encoder = src_tensor_2
                        
                        elif src2 == 'camera': 
                            encoder2 = src_tensor_2 #encoder2 is defined for camera, src_tensor_2 comes from src2
                            encoder = src_tensor_1 
                        else: 
                            encoder2 = src_tensor_2
                            encoder = src_tensor_1
                    f_src_1 = src_encoder(encoder).unsqueeze(dim = 0) #src tensor 1 equals camera tensor
                    f_src_2 = src_encoder2(encoder2).unsqueeze(dim = 0)
                    f_src = torch.cat((f_src_1,f_src_2),dim=0)

                else:
                    f_src = src_encoder(src_tensor).unsqueeze(dim = 0)

                f_tgt = tgt_encoder(tgt_tensor).unsqueeze(dim = 0)       
                fused = transformer(f_src,f_tgt).squeeze(dim = 0)   #takes src and tgt tensor in order -mg           
                f_tgt = f_tgt.squeeze(dim = 0) # squeeze tgt tensor

            
                if i%100 == 0:
                    # print(fused.shape,f_tgt.shape)
                    print('src',torch.isnan(f_src).any(),'transformer',torch.isnan(fused).any(),'tgt',torch.isnan(f_tgt).any())
                    # breakpoint()
                    reference = f_tgt.tolist() #take tgt tensor as reference -mg
                    candidate = fused.tolist()
                    distance = spatial.distance.cosine(reference[0], candidate[0])
                    #train_acc = 1 - distance / max(len(reference), len(candidate))
                    test_acc = 1 - distance
                    print('Epoch {} Iteration {}/{} Train_acc {}'.format(epoch,i,len(lidar_loader),test_acc))
            
                    tb_logger.add_scalar('test accuracy',test_acc,i)



