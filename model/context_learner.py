import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import math



class ContextNet(nn.Module):
    def __init__(self,cfg,num_channels, ful_con):
        super(ContextNet, self).__init__()

        self.out_dim = out_dim  = 32 
        self.ful_con = ful_con
                
        self.padding = SamePad2d(kernel_size=3, stride=1)

        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(num_channels, 128, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(128, eps=0.001)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32, eps=0.001)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(16, eps=0.001)
        self.conv4 = nn.Conv2d(16, 8, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(8, eps=0.001)
        self.conv5 = nn.Conv2d(8, num_channels,kernel_size=3, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.fc= nn.Linear(self.ful_con,256)  
        self.fc1=nn.Linear(256,out_dim)
        

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.maxpool(x) 
        x = self.bn4(x)
        x = self.relu(x) 
        x = self.conv5(x)
        x = self.maxpool(x)
        x = self.sigmoid(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc1(x)
        return x





############################################################
#  Pytorch Utility Functions
############################################################


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__