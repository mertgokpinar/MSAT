from cProfile import label
from dataclasses import dataclass
from sys import breakpointhook
from traceback import FrameSummary
from matplotlib.pyplot import locator_params
import torch, os, numpy as np, copy
import cv2
import glob
import pandas as pd
from radar import radar_polar_to_cartesian
import radiate
import sys
from .map import GeometricMap

from model.sequence_folders import SequenceFolder
from model.sequence_folders_lidar import SequenceFolder_lidar
from model.sequence_folders_mono import SequenceFolder_mono

import os
import numpy as np
from torch.utils.data import Dataset

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

from PIL import Image, ImageOps
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms, ToTensor
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

def load_undistorted_as_float(path):
    img = Image.open(path)
    img = img.resize((800, 445))
    img = ImageOps.grayscale(img) #reading image as gray scale since map encoder takes 1 channel tensor -mg
    return img

def preprocess_img(path):
    # Crop bottom
    offset_y = 100
    in_h = 900
    in_w = 1600
    (left, upper, right, lower) = (0, 0, in_w, in_h-offset_y)
    img = Image.open(path)
    img = img.crop((left, upper, right, lower))

    # Resize
    (width, height) = (320, 192)
    img = img.resize((width, height))
    img = ImageOps.grayscale(img)

    return img
class LiDARImageTransformer:
    def __init__(self, cart_pixels=512, max_range=50.0):
        self.cart_pixels = cart_pixels
        self.max_range = max_range
    def ptc2img(self, data):
        if data.shape[0] != 4:
            raise ValueError("Input must be [4,N]. Got {}".format(
                data.shape))
        power_sum, _, _ = np.histogram2d(
            x=data[0], y=data[1],
            bins=[self.cart_pixels, self.cart_pixels],
            weights=data[3], normed=False,
            range=[[-self.max_range, self.max_range],
                   [-self.max_range, self.max_range]]
        )
        power_count, _, _ = np.histogram2d(
            x=data[0], y=data[1],
            bins=[self.cart_pixels, self.cart_pixels],
            range=[[-self.max_range, self.max_range],
                   [-self.max_range, self.max_range]]
        )
        img = np.divide(
            power_sum, power_count,
            out=np.zeros_like(power_sum), where=power_count != 0
        )
        img = img.astype(np.float32)[np.newaxis, :, :]  # / 255.

        return img

class NuScenesDataset(Dataset):
    def __init__(self, nusc, nuscenes_root, scene_name):
        """
        Initialize the dataset with NuScenes root and the specific scene.
        """
        self.nusc = nusc 
        self.transform_lidar = LiDARImageTransformer(cart_pixels= 512,max_range = 80)
        self.scene_name = scene_name
        self.scene_tokens = [s['token'] for s in self.nusc.scene if s['name'] == scene_name]
        #self.scene_tokens = nusc.field2token('scene', 'name', scene_name)
        self.sample_tokens = []
        for scene_token in self.scene_tokens:
            scene = self.nusc.get('scene', scene_token)
            first_sample_token = scene['first_sample_token']
            self.sample_tokens.append(first_sample_token)
            next_token = self.nusc.get('sample', first_sample_token)['next']
            while next_token:
                self.sample_tokens.append(next_token)
                next_token = self.nusc.get('sample', next_token)['next']
    def __len__(self):
        """
        Return the number of samples in the scene.
        """
        return len(self.sample_tokens)

    def __getitem__(self, idx):
        """
        Return the front left camera image and lidar data as tensors for a given frame index.
        """
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', sample_token)
        
        # Front left camera data
        cam_front_left_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT_LEFT'])
        camera_image_path = os.path.join(self.nusc.dataroot, cam_front_left_data['filename'])
        image = load_undistorted_as_float(camera_image_path)
        #image = preprocess_img(camera_image_path)
        transform = transforms.Compose([transforms.PILToTensor()])
        image_tensor = transform(image) #transforms image to tensor -mg
        image_tensor = image_tensor.float() #context learner takes tensor as float -mg
        
        # Lidar data
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        pc = LidarPointCloud.from_file(lidar_path)
        lidar_img = self.transform_lidar.ptc2img(pc.points )
        lidar_tensor = torch.from_numpy(lidar_img).float()  # Convert LiDAR points to tensor
        
        return image_tensor, lidar_tensor

class preprocess(object):
    
    def __init__(self, nusc, data_root, seq_name, parser, log,split='train', phase='training'):
        self.parser = parser
        self.dataset = parser.dataset
        self.data_root = data_root
        self.past_frames = parser.past_frames
        self.future_frames = parser.future_frames
        self.frame_skip = parser.get('frame_skip', None)
        self.min_past_frames = parser.get('min_past_frames', self.past_frames)
        self.min_future_frames = parser.get('min_future_frames', self.future_frames)
        self.traj_scale = parser.traj_scale
        self.past_traj_scale = parser.traj_scale
        self.sensor=parser.sensor
        self.fix_origin = parser.get('fix_origin',False)
        #self.load_map = parser.get('load_map', False)
        #self.map_version = parser.get('map_version', '0.1')
        self.ro_params = {
            'cart_resolution': 0.25,
            'cart_pixels': 512,
            'rangeResolutionsInMeter':  0.175,
            'angleResolutionInRad': 0.015708,
            'radar_format': 'polar'
        }
    
        self.seq_name = seq_name
        self.sensor_radar = SequenceFolder(
            os.path.join(data_root,'sequences/'),
            transform=None, 
            seed=0,
            mode='train',
            sequence_length=3,
            skip_frames=1,
            dataset='radiate',
            ro_params=self.ro_params,
            load_camera=None,    
            cam_mode= 'mono',
            cam_transform=None,
            sequence=self.seq_name, 
            cam_preprocessed=1
            )

        self.sensor_lidar = SequenceFolder_lidar(os.path.join(data_root,'sequences/'),
            transform=None, 
            seed=0,
            lo_params = {
            'cart_pixels': 512},
            mode='train',
            sequence_length=3,
            skip_frames=1,
            dataset='radiate',
            load_camera=False,    
            cam_mode= 'mono',
            cam_transform=None,
            sequence=self.seq_name, 
            cam_preprocessed=False)
        
        self.sensor_camera= SequenceFolder_mono(os.path.join(data_root,'sequences/'),dataset='radiate', 
                 seed=None, train=True,
                 sequence_length=3, transform=None, skip_frames=1, preprocessed=True,
                 sequence=self.seq_name)
        
        if parser.dataset == 'nuscenes_pred':
         self.nuscenes_loader = NuScenesDataset(nusc= nusc, scene_name= seq_name, nuscenes_root= '/home/mide/Desktop/nuscenes')

        #we are calling sequence folder when we initialize preprocessor to save from computing time. -mg
        self.split = split
        self.phase = phase
        self.log = log
        
        
        if parser.dataset == 'nuscenes_pred':
            label_path = os.path.join(data_root, 'label/{}/{}.txt'.format(split, seq_name))
            delimiter = ' '
        elif parser.dataset in {'radiate'}:
            delimiter=''
            seq=radiate.Sequence(os.path.join(data_root, 'sequences/{}'.format(seq_name)))
            #self.seq=seq
            label_path= os.path.join(data_root,'label_for_2s_prediction/{}.txt'.format(seq_name))   #changed labels for 2s prediction 25/03/2024 -mg
            sequence_path=os.path.join(data_root, 'sequences/{}/Navtech_Cartesian.txt'.format(seq_name)) # gets sequence root path for sensor. -mg
        else:
            assert False, 'error'

        self.gt = np.genfromtxt(label_path, delimiter=delimiter, dtype=str)
        if self.dataset == 'nuscenes_pred':
            self.class_names = class_names = {'Pedestrian': 1, 'Car': 2, 'Cyclist': 3, 'Truck': 4, 'Van': 5, 'Tram': 6, 'Person': 7, \
                'Misc': 8, 'DontCare': 9, 'Traffic_cone': 10, 'Construction_vehicle': 11, 'Barrier': 12, 'Motorcycle': 13, \
                'Bicycle': 14, 'Bus': 15, 'Trailer': 16, 'Emergency': 17, 'Construction': 18}
            for row_index in range(len(self.gt)):
                self.gt[row_index][2] = class_names[self.gt[row_index][2]]
        self.gt = self.gt.astype('float32')
        print('self.gt_shape',np.shape(self.gt))

        frame = self.gt[:, 0].astype(np.float32).astype(np.int)
        
        #frame= seq.load_timestamp(sequence_path) #returns list of all timestamps in dict format -mg
        fr_start, fr_end = np.min(frame), np.max(frame) #frames stored in dict['frames'] -mg
        print(fr_start)
        self.init_frame = fr_start
        self.num_fr = fr_end-1   #frame data -mg
        print(self.num_fr)

        self.xind, self.zind = 13, 15 #what is self.zind and self.xind

    def load_sensors(self,frame):
     
        
        # data_root = '/media/otonom/cold_storage/radiate'
        # radar_frame=(frame/self.num_fr)*int(self.num_fr/30)
        # radar_frame=int(radar_frame)*30

        if self.sensor=='all':
            lidar=self.sensor_lidar.__getitem__(frame)
            camera=self.sensor_camera.__getitem__(frame)
            radar=self.sensor_radar.__getitem__(frame)
            return lidar, camera , radar
        else:
            if self.sensor=='lidar':
                tensor=self.sensor_lidar.__getitem__(frame)
            elif self.sensor=='camera':
                tensor= self.sensor_camera.__getitem__(frame)
            else:     
                tensor=self.sensor_radar.__getitem__(frame)  # [1x512x512] [CxHxW] -> [BxCxHxW]

            return tensor

    '''
    load_sensors takes sequence name , frame of interest and total frame number of sequence.
    /////////////////////////////
    SequenceFolder takes specific sequence name it does not read all sequence folders, also 
    sample_shuffling is disabled with in class, __getitem__ returns only 'tgt' image tensors. 

    '''

    def GetID(self, data):
        id = []
        for i in range(data.shape[0]):
            id.append(data[i, 1].copy())
        return id

    def TotalFrame(self):
        return self.num_fr

    def PreData(self, frame):
        DataList = []
        # seq=self.seq
        for i in range(self.past_frames):
            if frame - i < self.init_frame:              
                data = []
            data = self.gt[self.gt[:, 0] == (frame - i * self.frame_skip)]
            DataList.append(data)

        # print('Frame',frame)#debug
        # print('Pre_Frames',DataList)#debug
        # breakpoint()
        return DataList
    
    def FutureData(self, frame):
        DataList = []
        # seq=self.seq
        for i in range(1, self.future_frames + 1):
            data = self.gt[self.gt[:, 0] == (frame + i * self.frame_skip)]
            DataList.append(data)

        # print('Frame',frame) #debug
        # print('Fut_Frames',DataList)#debug
        # breakpoint()
        return DataList

    def get_valid_id(self, pre_data, fut_data):
        cur_id = self.GetID(pre_data[0])
        valid_id = []
        for idx in cur_id:
           
            if self.dataset == 'nuscenes_pred':
                exist_pre = [(False if isinstance(data, list) else (idx in data[:, 1])) for data in pre_data[:self.min_past_frames]] #ayni idye sahip arac min past ve min future framelerda var mi diye bakiyor. -mg            
                exist_fut = [(False if isinstance(data, list) else (idx in data[:, 1])) for data in fut_data[:self.min_future_frames]]
            else:
                exist_pre = [(False if data[0,1] ==-1 else (idx in data[:, 1])) for data in pre_data[:self.min_past_frames]] #ayni idye sahip arac min past ve min future framelerda var mi diye bakiyor. -mg            
                exist_fut = [(False if data[0,1]==-1 else (idx in data[:, 1])) for data in fut_data[:self.min_future_frames]] 

            # print(exist_pre)
            # breakpoint()

           
            if np.all(exist_pre) and np.all(exist_fut):
                valid_id.append(idx) #how valid id is determined ? =mg
        return valid_id

    def get_pred_mask(self, cur_data, valid_id):
        pred_mask = np.zeros(len(valid_id), dtype=np.int)
        for i, idx in enumerate(valid_id):
            pred_mask[i] = cur_data[cur_data[:, 1] == idx].squeeze()[-1]
        return pred_mask

    def get_heading(self, cur_data, valid_id):
        heading = np.zeros(len(valid_id))
        for i, idx in enumerate(valid_id):
            heading[i] = cur_data[cur_data[:, 1] == idx].squeeze()[16]
        return heading

    # def load_scene_map(self):
    #     map_file = f'{self.data_root}/map_{self.map_version}/{self.seq_name}.png'
    #     map_vis_file = f'{self.data_root}/map_{self.map_version}/vis_{self.seq_name}.png'
    #     map_meta_file = f'{self.data_root}/map_{self.map_version}/meta_{self.seq_name}.txt'
    #     self.scene_map = np.transpose(cv2.imread(map_file), (2, 0, 1))
    #     self.scene_vis_map = np.transpose(cv2.cvtColor(cv2.imread(map_vis_file), cv2.COLOR_BGR2RGB), (2, 0, 1))
    #     self.meta = np.loadtxt(map_meta_file)
    #     self.map_origin = self.meta[:2]
    #     self.map_scale = scale = self.meta[2]
    #     homography = np.array([[scale, 0., 0.], [0., scale, 0.], [0., 0., scale]]) #scale comes from map meta data
    #     self.geom_scene_map = GeometricMap(self.scene_map, homography, self.map_origin) #self.map_origin comes from map meta file we can manually add them.
    #     self.scene_vis_map = GeometricMap(self.scene_vis_map, homography, self.map_origin) 

    def PreMotion(self, DataTuple, valid_id):
        motion = []
        mask = []
        for identity in valid_id:
            mask_i = torch.zeros(self.past_frames)
            box_3d = torch.zeros([self.past_frames, 2])
            for j in range(self.past_frames):
                past_data = DataTuple[j]              # past_data
                if len(past_data) > 0 and identity in past_data[:, 1]:
                    found_data = past_data[past_data[:, 1] == identity].squeeze()[[self.xind, self.zind]] / self.past_traj_scale
                    box_3d[self.past_frames-1 - j, :] = torch.from_numpy(found_data).float()
                    mask_i[self.past_frames-1 - j] = 1.0
                elif j > 0:
                    box_3d[self.past_frames-1 - j, :] = box_3d[self.past_frames - j, :]    # if none, copy from previous
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(box_3d)
            mask.append(mask_i)
        return motion, mask

    def FutureMotion(self, DataTuple, valid_id):
        motion = []
        mask = []
        for identity in valid_id:
            mask_i = torch.zeros(self.future_frames)
            pos_3d = torch.zeros([self.future_frames, 2])
            for j in range(self.future_frames):
                fut_data = DataTuple[j]              # cur_data
                if len(fut_data) > 0 and identity in fut_data[:, 1]:
                    # print('actual_data:',fut_data[fut_data[:, 1] == identity].squeeze()[[self.xind, self.zind]]) 
                    found_data = fut_data[fut_data[:, 1] == identity].squeeze()[[self.xind, self.zind]] / self.traj_scale
                    # print('found_data:',found_data) #debug
                    pos_3d[j, :] = torch.from_numpy(found_data).float()
                    
                    # breakpoint()
                    mask_i[j] = 1.0
                elif j > 0:
                    pos_3d[j, :] = pos_3d[j - 1, :]    # if none, copy from previous
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(pos_3d)
            mask.append(mask_i)
        return motion, mask

    def __call__(self, frame):

        assert frame - self.init_frame >= 0 and frame - self.init_frame <= self.TotalFrame() - 1, 'frame is %d, total is %d' % (frame, self.TotalFrame())

        pre_data = self.PreData(frame)
        fut_data = self.FutureData(frame)   
        valid_id = self.get_valid_id(pre_data, fut_data)

        if self.sensor == 'all' and self.dataset=='radiate':
            lidar,camera,radar = self.load_sensors(frame)
        elif self.sensor != 'all' and self.dataset=='radiate':
            sensor = self.load_sensors(frame)
        elif self.dataset == 'nuscenes_pred':
            camera, lidar = self.nuscenes_loader.__getitem__(frame)
            radar = None
            if self.sensor == 'camera':
                sensor = camera
            elif self.sensor == 'lidar':
                sensor = lidar
            else:
                sensor = None
        if self.sensor != 'all':
            
            lidar=None
            camera=None
            radar=None    

        # dataloader will return none if sample dont have pre and fut frames and no agents to predict in frame
        # batch size is fixed to 20, maximum number of predicted agents will be 20
        if len(pre_data[0]) == 0 or len(fut_data[0]) == 0 or len(valid_id) == 0 or len(valid_id)>=20:
            return None

        if self.dataset == 'radiate' or 'nuscenes_pred': 
            pred_mask = self.get_pred_mask(pre_data[0], valid_id)
            heading = self.get_heading(pre_data[0], valid_id)
        else:
            pred_mask = None
            heading = None

        pre_motion_3D, pre_motion_mask = self.PreMotion(pre_data, valid_id)
        fut_motion_3D, fut_motion_mask = self.FutureMotion(fut_data, valid_id)

        if self.fix_origin:

            pre_motion_3D = torch.stack(pre_motion_3D,dim=0)
            fut_motion_3D = torch.stack(fut_motion_3D,dim=0)
            fut_motion_3D = fut_motion_3D - pre_motion_3D[0][0]
            pre_motion_3D = pre_motion_3D - pre_motion_3D[0][0]
            fut_motion_3D = list(fut_motion_3D)
            pre_motion_3D = list(pre_motion_3D)
             
        data = {
            'pre_motion_3D': pre_motion_3D,
            'fut_motion_3D': fut_motion_3D,
            'fut_motion_mask': fut_motion_mask,
            'pre_motion_mask': pre_motion_mask,
            'pre_data': pre_data,
            'fut_data': fut_data,
            'heading': heading,
            'valid_id': valid_id,
            'traj_scale': self.traj_scale,
            'pred_mask': pred_mask,
            #'scene_map': self.geom_scene_map,
            'seq': self.seq_name,
            'frame': frame,
            'sensors': sensor, #mg
            'lidar' : lidar,
            'camera' : camera,
            'radar' : radar
        }

        #print(sys.getsizeof(datak)) #debug
        return data
