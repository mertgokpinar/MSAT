#TODO merge feature vectors with different batchsizes using cat.
#then feed new feature vector into fully connected layer 
#make sure you extract features of all sensor on the fly.

from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn

class PoseFusionNet(nn.Module):

    def __init__(self,batch_size):
        super(PoseFusionNet, self).__init__()

        
        self.fc = nn.Sequential(
            nn.Linear(ful_con, 64),
            nn.ReLU(True),
            nn.Linear(64, 32)
        )
      

    def forward(self, vo):
       
        # vo = torch.flatten(vo, dim=1)
        vo_pose = self.fc(vo)

      

        return vo_pose
