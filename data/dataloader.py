
from data.nuscenes_pred_split import get_nuscenes_pred_split
import os, random, numpy as np, copy
from .preprocessor import preprocess
from .ethucy_split import get_ethucy_split
from utils.utils import print_log
from nuscenes.nuscenes import NuScenes

def open_sensor(data_root): #mg
    train=[]
    val=[]
    test=[]
    with open (os.path.join(data_root, 'train.txt')) as file:
        for line in file:
            train.append(line.strip())

    with open (os.path.join(data_root, 'val.txt')) as file:
        for line in file:
            val.append(line.strip())

    with open (os.path.join(data_root, 'test.txt')) as file:
        for line in file:
            test.append(line.strip())

    return train,val,test


class data_generator(object): # this data_generator reads our ground truth data, annotations in radiate dataset and returns current index object
   
    
    def __init__(self, parser,log, split='train', phase='training'):
        self.past_frames = parser.past_frames
        self.min_past_frames = parser.min_past_frames
        self.frame_skip = parser.get('frame_skip', None)
        self.phase = phase
        self.split = split
        assert phase in ['training', 'testing'], 'error'
        assert split in ['train', 'val', 'test'], 'error'

        if parser.dataset == 'nuscenes_pred':
            data_root = parser.data_root_nuscenes_pred           
            seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root) #creates sequence from dataset folders.
            self.init_frame = 0
            self.nusc = NuScenes(version='v1.0-trainval', dataroot= '/media/nfs/nuscenes', verbose=True)
        # elif parser.dataset in {'eth', 'hotel', 'univ', 'zara1', 'zara2'}:
        #     data_root = parser.data_root_ethucy            
        #     seq_train, seq_val, seq_test = get_ethucy_split(parser.dataset)
        #     self.init_frame = 0
        elif parser.dataset== 'radiate':
             self.nusc=None
             data_root = parser.data_root_radiate
             seq_train, seq_val, seq_test = open_sensor(data_root) #takes train and test sequence .txt files. -mg
             
              
             self.init_frame = 0
        else:
            raise ValueError('Unknown dataset!')
        #we will read our data set in here and pass it to msat

        process_func = preprocess #preprocess returns data{} ,for ground data  

        # we need to load_sensor in this line of code
        self.data_root = os.path.join(data_root,'sequences/')

        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        if self.split == 'train':  self.sequence_to_load = seq_train
        elif self.split == 'val':  self.sequence_to_load = seq_val
        elif self.split == 'test': self.sequence_to_load = seq_test
        else:                      assert False, 'error'
        self.num_total_samples = 0
        self.num_sample_list = []
        self.sequence = []
        for seq_name in self.sequence_to_load: #creates data sequence from dataset. -mg
            none_data=0

            print_log("loading sequence {} ...".format(seq_name), log=log)
            preprocessor = process_func(self.nusc, data_root, seq_name, parser, log, self.split, self.phase) 

            #provides data_root to preprocess.py so it can load data -mg
            #we will read our data set in here and pass it to msat -mg

            num_seq_samples = preprocessor.num_fr - (parser.min_past_frames + parser.min_future_frames - 1) * self.frame_skip
            
            self.num_total_samples += num_seq_samples
            self.num_sample_list.append(num_seq_samples)
            self.sequence.append(preprocessor) #sequence list for ground truth data.

            # for i in range(parser.min_past_frames+2,num_seq_samples):
            #     data = preprocessor(i)
            #     if data is None:
            #         #print('ok')
            #         none_data+=1
            # print(f'number of samples in sequence: {(num_seq_samples-none_data)}')
 
        self.sample_list = list(range(self.num_total_samples))
        self.index = 0
        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)


    
    def shuffle(self):
        random.shuffle(self.sample_list)

    # TODO: make sure frame-id and seq-id are correct 
    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):    # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                frame_index = index_tmp + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame     # from 0-indexed list index to 1-indexed frame index (for mot)
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def is_epoch_end(self):
        if self.index >= self.num_total_samples:
            self.index = 0      # reset
            return True
        else:
            return False
        

    def next_sample(self):
        sample_index = self.sample_list[self.index]
        seq_index, frame = self.get_seq_and_frame(sample_index)
        seq = self.sequence[seq_index]
        self.index += 1
        data = seq(frame)
        return data     

    def __call__(self):
        return self.next_sample()
