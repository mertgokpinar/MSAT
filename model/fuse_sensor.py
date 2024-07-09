import torch 
from torch import nn 
import os 
import random 
import numpy as np
from .fuse_network import Transformer

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)

class FuseModule(nn.Module):

    def __init__(self,path):
        super(FuseModule, self).__init__()

        self.model = Transformer(32,dropout=0)
        self.model.load_state_dict(torch.load(path))
        for param in self.model.parameters():
            param.requires_grad = False

        print('loaded weights from:', path)

        #self.model.eval()
        
    def forward(self, src,tgt):
        return self.model(src,tgt)