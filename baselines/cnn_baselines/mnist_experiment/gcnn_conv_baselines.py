import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M, P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling

class AllConvNet1(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet1, self).__init__()
        self.conv1 = P4ConvZ2(input_size, 96, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(96) 
        self.conv2 = P4ConvP4(96, 96, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(96)
        self.conv3 = P4ConvP4(96, 96, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm3d(96)
        self.conv4 = P4ConvP4(96, 192, 3, padding=1)
        self.bn4 = nn.BatchNorm3d(192)
        self.conv5 = P4ConvP4(192, 192, 3, padding=1)
        self.bn5 = nn.BatchNorm3d(192)
        self.conv6 = P4ConvP4(192, 192, 3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm3d(192)
        self.conv7 = P4ConvP4(192, 192, 3, padding=1)
        self.bn7 = nn.BatchNorm3d(192)
        self.conv8 = P4ConvP4(192, 192, 1)
        self.bn8 = nn.BatchNorm3d(192)
        self.hidden_fc = nn.Linear(192*4*7*7,128)
        self.class_fc = nn.Linear(128, n_classes)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.bn1(self.conv1(x_drop)))
        conv2_out = F.relu(self.bn2(self.conv2(conv1_out)))
        conv3_out = F.relu(self.bn3(self.conv3(conv2_out)))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.bn4(self.conv4(conv3_out_drop)))
        conv5_out = F.relu(self.bn5(self.conv5(conv4_out)))
        conv6_out = F.relu(self.bn6(self.conv6(conv5_out)))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.bn7(self.conv7(conv6_out_drop)))
        conv8_out = F.relu(self.bn8(self.conv8(conv7_out)))
        conv8_out = conv8_out.view(-1,192*4*7*7)
        hidden_out = F.relu(self.hidden_fc(conv8_out)) 
        class_out = F.relu(self.class_fc(hidden_out)) 
        return class_out

class AllConvNet2(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet2, self).__init__()
        self.conv1 = P4MConvZ2(input_size, 96, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(96) 
        self.conv2 = P4MConvP4M(96, 96, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(96)
        self.conv3 = P4MConvP4M(96, 96, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm3d(96)
        self.conv4 = P4MConvP4M(96, 192, 3, padding=1)
        self.bn4 = nn.BatchNorm3d(192)
        self.conv5 = P4MConvP4M(192, 192, 3, padding=1)
        self.bn5 = nn.BatchNorm3d(192)
        self.conv6 = P4MConvP4M(192, 192, 3, padding=1, stride=2)
        self.bn6 = nn.BatchNorm3d(192)
        self.conv7 = P4MConvP4M(192, 192, 3, padding=1)
        self.bn7 = nn.BatchNorm3d(192)
        self.conv8 = P4MConvP4M(192, 192, 1)
        self.bn8 = nn.BatchNorm3d(192)
        self.hidden_fc = nn.Linear(192*8*7*7,128)
        self.class_fc = nn.Linear(128, n_classes)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.bn1(self.conv1(x_drop)))
        conv2_out = F.relu(self.bn2(self.conv2(conv1_out)))
        conv3_out = F.relu(self.bn3(self.conv3(conv2_out)))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.bn4(self.conv4(conv3_out_drop)))
        conv5_out = F.relu(self.bn5(self.conv5(conv4_out)))
        conv6_out = F.relu(self.bn6(self.conv6(conv5_out)))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.bn7(self.conv7(conv6_out_drop)))
        conv8_out = F.relu(self.bn8(self.conv8(conv7_out)))
        conv8_out = conv8_out.view(-1,192*8*7*7)
        hidden_out = F.relu(self.hidden_fc(conv8_out)) 
        class_out = F.relu(self.class_fc(hidden_out)) 
        return class_out

def SimpleConvP4(): 
    return AllConvNet1(1) 

def SimpleConvP4M():
    return AllConvNet2(1)
