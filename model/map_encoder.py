import torch
import torch.nn as nn
from .context_learner import ContextNet

class MapEncoder(nn.Module):
    def __init__(self, cfg, ful_con):
        super().__init__()
        model_id = cfg.get('model_id', 'context_net')
        dropout = cfg.get('dropout', 0.0)
        self.normalize = cfg.get('normalize', True)
        self.dropout = nn.Dropout(dropout)
        if model_id == 'context_net':  
            self.model = ContextNet(cfg, num_channels=1, ful_con=ful_con) 
            self.out_dim = self.model.out_dim
        
                  
        else:
            raise ValueError('unknown map encoder!')
       
    def forward(self, x):
        if self.normalize:
            x = x * 2. - 1.
        x = self.model(x)
        x = self.dropout(x)
        return x
