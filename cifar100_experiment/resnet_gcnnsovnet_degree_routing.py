import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim 
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M, P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = P4ConvP4(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = P4ConvP4(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                P4ConvP4(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetPreCapsule(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNetPreCapsule, self).__init__()
        self.in_planes = 64

        self.conv1 = P4ConvZ2(3, 64, kernel_size=3, stride=1, padding=1, bias=False)#(b_size,64,32,32)
        self.bn1 = nn.BatchNorm3d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)#(b_size,64,32,32)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)#(b_size,128,16,16)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out

class PrimaryCapsules(nn.Module):
    def __init__(self,in_channels,num_capsules,out_dim,H=16,W=16):
        super(PrimaryCapsules,self).__init__()
        self.in_channels = in_channels
        self.num_capsules = num_capsules
        self.out_dim = out_dim
        self.H = H
        self.W = W
        self.preds = nn.Sequential(
                                   P4ConvP4(in_channels,num_capsules*out_dim,kernel_size=1),
                                   nn.LayerNorm((num_capsules*out_dim,4,H,W)))

    def forward(self,x):
        primary_capsules = self.preds(x)
        primary_capsules = primary_capsules.view(-1,self.num_capsules,self.out_dim,4,self.H,self.W)
        return primary_capsules

class ConvCapsule(nn.Module):
    def __init__(self,in_caps,in_dim,out_caps,out_dim,kernel_size,stride,padding):
        super(ConvCapsule,self).__init__()
        self.in_caps = in_caps
        self.in_dim = in_dim
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.preds = nn.Sequential(
                                   P4ConvP4(in_dim,out_caps*out_dim,kernel_size=kernel_size,stride=stride,padding=padding),
                                   nn.BatchNorm3d(out_caps*out_dim))
     
    def forward(self,in_capsules):
        batch_size, _, _, _, H, W = in_capsules.size()
        in_capsules = in_capsules.view(batch_size*self.in_caps,self.in_dim,4,H,W)
        predictions = self.preds(in_capsules)
        _,_,_, H, W = predictions.size()
        predictions = predictions.view(batch_size, self.in_caps, self.out_caps*self.out_dim, 4, H, W)
        predictions = predictions.view(batch_size, self.in_caps, self.out_caps, self.out_dim, 4, H, W)
        out_capsules = self.degree_routing(predictions)
        return out_capsules

    def squash(self, inputs, dim):
        norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
        scale = norm**2 / (1 + norm**2) / (norm + 1e-8)
        return scale * inputs

    def cosine_similarity(self,predictions,eps=1e-8):
        dot_product = torch.matmul(predictions,predictions.transpose(-1,-2))
        norm_sq = torch.norm(predictions,dim=-1,keepdim=True)**2 
        eps_matrix = eps*torch.ones_like(norm_sq)
        norm_sq = torch.max(norm_sq,eps_matrix)
        similarity_matrix = dot_product/norm_sq
        return similarity_matrix

    def degree_routing(self,predictions):
        batch_size,_,_,_,_, H, W = predictions.size()
        predictions_permute = predictions.permute(0,2,4,5,6,1,3)#(batch_size,num_out_capsules,4,H,W,num_in_capsules,out_capsule_dim)
        affinity_matrices = self.cosine_similarity(predictions_permute)    
        degree_score = F.softmax(torch.sum(affinity_matrices,dim=-1,keepdim=True),dim=5)#(batch_size,num_out_capsules,4,H,W,num_in_capsules,1)
        degree_score = (degree_score).permute(0,5,1,6,2,3,4)#(batch_size,num_in_capsules,num_out_capsules,1,4,H,W)
        s_j = (degree_score * predictions).sum(dim=1)
        v_j = self.squash(s_j,dim=3)
        return v_j.squeeze(dim=1)

class ResnetGcnnsovnetDegreeRouting(nn.Module):
    def __init__(self):
        super(ResnetGcnnsovnetDegreeRouting,self).__init__()
        self.resnet_precaps = ResNetPreCapsule(BasicBlock,[3,4]) 
        self.primary_caps = PrimaryCapsules(128,32,16,16,16)#for cifar10, H,W = 16, 16. For MNIST etc. H,W = 14,14.
        self.conv_caps1 = ConvCapsule(in_caps=32,in_dim=16,out_caps=32,out_dim=16,kernel_size=3,stride=2,padding=0)
        self.conv_caps2 = ConvCapsule(in_caps=32,in_dim=16,out_caps=32,out_dim=16,kernel_size=3,stride=1,padding=0)
        self.conv_caps3 = ConvCapsule(in_caps=32,in_dim=16,out_caps=32,out_dim=16,kernel_size=3,stride=1,padding=0)
        self.class_caps = ConvCapsule(in_caps=32,in_dim=16,out_caps=100,out_dim=16,kernel_size=3,stride=1,padding=0)
        self.linear = nn.Linear(16,1)
        

    def forward(self,x):
        conv_output = self.resnet_precaps(x)
        primary_caps = self.primary_caps(conv_output)
        conv_caps1 = self.conv_caps1(primary_caps)
        conv_caps2 = self.conv_caps2(conv_caps1)
        conv_caps3 = self.conv_caps3(conv_caps2)
        class_capsules = self.class_caps(conv_caps3)
        class_capsules = class_capsules.squeeze()
        class_capsules_norm = torch.norm(class_capsules,dim=2,keepdim=False)
        max_length_indices_per_type = torch.max(class_capsules_norm,dim=2)[1]  
        masked = class_capsules.new_tensor(torch.eye(class_capsules.size(3)))
        masked = masked.index_select(dim=0, index=max_length_indices_per_type.data.view(-1))
        masked = masked.view(class_capsules.size(0),class_capsules.size(1),-1).unsqueeze(2)
        class_capsules_output = (class_capsules*masked).sum(3)
        class_predictions = self.linear(class_capsules_output).squeeze()
        return class_predictions
