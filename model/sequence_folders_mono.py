from cv2 import transform
import torch.utils.data as data
import numpy as np
#from imageio import imread
from pathlib import Path
import random
import torchvision.transforms as transforms

from PIL import Image, ImageOps
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from .camera_model import CameraModel

import utils_warp as utils


class SequenceFolder_mono(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, dataset='radiate', seed=None, train=True,
                 sequence_length=3, transform=None, skip_frames=1, preprocessed=True,
                 sequence=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        self.dataset = dataset
        if sequence is not None:
            self.scenes = [self.root/sequence]
        else:
            scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
            self.scenes = [self.root/folder.strip()
                           for folder in open(scene_list_path) if not folder.strip().startswith("#")]
        self.transform = transform
        self.train = train
        self.k = skip_frames
        self.preprocessed = preprocessed

        if dataset == 'radiate':
            if self.preprocessed:
                self.mono_folder = 'stereo_undistorted/left'
            else:
                self.mono_folder = 'zed_left'
        elif dataset == 'robotcar':
            if self.preprocessed:
                self.mono_folder = 'stereo_undistorted/left'
            else:
                self.mono_folder = 'stereo/left'
                self.cam_model = CameraModel()
        elif dataset == 'cadcd':
            if self.preprocessed:
                self.mono_folder = 'raw/image_07/cam_preprocessed'
            else:
                self.mono_folder = 'raw/image_07/data'
        else:
            raise NotImplementedError(
                'The chosen dataset is not implemented yet! Given: {}'.format(dataset))

        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        # k skip frames
        
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k,
                            demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            print('camera_scene_name:',scene) #debug code delet this
            # intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            intrinsics = utils.get_intrinsics_matrix(
                self.dataset, preprocessed=self.preprocessed)
            #imgs = sorted(scene.files('*.png'))
            imgs = sorted(list((scene/self.mono_folder).glob('*.png')))

            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                sample = {'intrinsics': [],
                          'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                    sample['intrinsics'].append(intrinsics)
                sequence_set.append(sample)

        # if self.train:
        #     random.shuffle(sequence_set)

        self.samples = sequence_set

    def load_as_float(self, path):
        # img = imread(path).astype(np.float32)
        img = Image.open(path)
        # img = img.convert("RGB")
        # img = np.array(img)
        if self.dataset == 'robotcar':
            img = demosaic(img, 'gbrg')
            img = self.cam_model.undistort(img)
        img = np.array(img).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    def load_undistorted_as_float(self, path):
        img = Image.open(path)
        img = ImageOps.grayscale(img) #reading image as gray scale since map encoder takes 1 channel tensor -mg
        return img

    
    def __getitem__(self, index):
        sample = self.samples[index]
        if self.preprocessed or self.dataset == 'radiate' or self.dataset == 'cadcd':
            # TODO: On-the-fly rectification support for RADIATE dataset
            tgt_img = self.load_undistorted_as_float(sample['tgt'])
            transform = transforms.Compose([transforms.PILToTensor()])
            tgt_img = transform(tgt_img) #transforms tgt image to tensor -mg
            tgt_img = tgt_img.float() #context learner takes tensor as float -mg
            # print('tgt_img_size:',tgt_img.size())
            # breakpoint()
            ref_imgs = [self.load_undistorted_as_float(ref_img)
                        for ref_img in sample['ref_imgs']]
        else:
            tgt_img = self.load_as_float(sample['tgt'])
            ref_imgs = [self.load_as_float(ref_img)
                        for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform(
                [tgt_img] + ref_imgs, [np.copy(i) for i in sample['intrinsics']])
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = [np.copy(i) for i in sample['intrinsics']]
        return tgt_img #, ref_imgs, intrinsics

    def __len__(self):
        return len(self.samples)
