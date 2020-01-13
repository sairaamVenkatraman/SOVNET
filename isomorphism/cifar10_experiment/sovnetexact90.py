import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim 
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M, P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision

class InitialP4Conv(nn.Module):
  def __init__(self):
      super(InitialP4Conv,self).__init__()
      self.conv = nn.Sequential(
                                P4ConvZ2(3,32,5),
                                nn.BatchNorm3d(32),
                                nn.SELU(),
                                P4ConvP4(32,32,5),
                                nn.BatchNorm3d(32),
                                nn.SELU(), 
                                P4ConvP4(32,32,3),
                                nn.BatchNorm3d(32),
                                nn.SELU()   
                               )
  def forward(self,x):
      out = self.conv(x)     
      return out                   

class PrimaryCapsules(nn.Module):
  '''Use 2d convolution to extract capsules'''
  def __init__(self,num_capsules=10,in_channels=32,out_channels=32):
    super(PrimaryCapsules, self).__init__()
    self.num_capsules = num_capsules
    self.out_channels = out_channels
    self.capsules = nn.Sequential(
                                  P4ConvP4(in_channels,out_channels*num_capsules,3),
                                  nn.BatchNorm3d(out_channels*num_capsules),
                                  nn.SELU(),
                                 )   
       
  def squash(self,x,dim):
    norm_squared = (x ** 2).sum(dim, keepdim=True)
    part1 = norm_squared / (1 +  norm_squared)
    part2 = x / torch.sqrt(norm_squared+ 1e-16)
    output = part1 * part2 
    return output
  
  def forward(self, x):
    output = self.capsules(x)
    H, W = output.size(-2), output.size(-1)
    output = output.view(-1,self.num_capsules,self.out_channels,4,H,W)
    output = self.squash(output,dim=2)    
    return output

class ConvolutionalCapsules(nn.Module):
  '''
      A capsule layer that uses one bottleneck layer per capsule-type
  '''
  def __init__(self,num_in_capsules,in_capsule_dim,num_out_capsules,out_capsule_dim,kernel_size,stride=1,padding=0):
    super(ConvolutionalCapsules,self).__init__()
    self.num_in_capsules = num_in_capsules
    self.in_capsule_dim = in_capsule_dim
    self.num_out_capsules = num_out_capsules
    self.out_capsule_dim = out_capsule_dim
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.projection_network = nn.Sequential(
                                            P4ConvP4(in_capsule_dim,out_capsule_dim*num_out_capsules,kernel_size,stride,padding),
                                            nn.BatchNorm3d(out_capsule_dim*num_out_capsules)
                                           )  
     
  def squash(self,x,dim):
    norm_squared = (x ** 2).sum(dim, keepdim=True)
    part1 = norm_squared / (1 +  norm_squared)
    part2 = x / torch.sqrt(norm_squared+ 1e-16)
    output = part1 * part2 
    return output
   
  def cosine_similarity(self,predictions,eps=1e-8):
    dot_product = torch.matmul(predictions,predictions.transpose(-1,-2))
    norm_sq = torch.norm(predictions,dim=-1,keepdim=True)**2 
    eps_matrix = eps*torch.ones_like(norm_sq)
    norm_sq = torch.max(norm_sq,eps_matrix)
    similarity_matrix = dot_product/norm_sq
    return similarity_matrix    

  def degree_routing(self,capsules):
    batch_size = capsules.size(0)
    grid_size = [capsules.size(-3),capsules.size(-2),capsules.size(-1)]
    ### Prediction u_hat ####
    capsules = capsules.view(batch_size*self.num_in_capsules,self.in_capsule_dim,grid_size[0],grid_size[1],grid_size[2])                 
    u_hat = self.projection_network(capsules)
    grid_size = [u_hat.size(-3),u_hat.size(-2),u_hat.size(-1)]
    u_hat = u_hat.view(batch_size,self.num_in_capsules,self.num_out_capsules,self.out_capsule_dim,grid_size[0],grid_size[1],grid_size[2])   
    ### u_hat:(batch_size,num_in_capsules,num_out_capsules,out_capsule_dim,4,H,W)
    u_hat_permute = u_hat.permute(0,2,4,5,6,1,3)#(batch_size,num_out_capsules,4,H,W,num_in_capsules,out_capsule_dim)
    affinity_matrices = self.cosine_similarity(u_hat_permute)    
    degree_score = F.softmax(torch.sum(affinity_matrices,dim=-1,keepdim=True),dim=5)#(batch_size,num_out_capsules,4,H,W,num_in_capsules,1)
    degree_score = (degree_score).permute(0,5,1,6,2,3,4)#(batch_size,num_in_capsules,num_out_capsules,1,4,H,W)
    s_j = (degree_score * u_hat).sum(dim=1)
    degree_score = degree_score.squeeze(3)
    v_j = self.squash(s_j,dim=3)
    return v_j.squeeze(dim=1), degree_score 

  def forward(self,capsules,routing_iterations=1):
    '''
        Input: (batch_size, num_in_capsules, in_capsule_dim, H, W)
        Output: (batch_size, num_out_capsules, out_capsule_dim, H', W')
    '''
    out_capsules = self.degree_routing(capsules)
    return out_capsules

class ReconstructionLayer(nn.Module):
  '''A layer to reconstruct the image from the last capsule layer'''
  def __init__(self,num_capsules,capsule_dim,im_size,im_channels):
    super(ReconstructionLayer,self).__init__()
    self.num_capsules = num_capsules
    self.capsule_dim = capsule_dim
    self.im_size = im_size
    self.im_channels = im_channels
    self.grid_size = 3
    self.FC = nn.Sequential(
                            nn.Linear(capsule_dim * num_capsules, num_capsules *(self.grid_size**2) ),
                            nn.SELU()
              )
    self.decoder = nn.Sequential(
                                 nn.ConvTranspose2d(in_channels=self.num_capsules, out_channels=32, kernel_size=5, stride=2),
                                 nn.SELU(),
                                 nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
                                 nn.SELU(),
                                 nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
                                 nn.SELU(),
                                 nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
                                 nn.SELU(),
                                 nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
                                 nn.SELU(),
                                 nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
                                 nn.SELU(),
                                 nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=5, stride=1),  
                                 nn.Sigmoid()    
                   )
  def forward(self, x, target=None):
      '''
          Input: (batch_size,num_capsules,capsule_dim,1,1)
          Output: (batch_size,im_channels,im_size,im_size)
      '''
      batch_size, num_capsules, out_channels,_,_ = x.size()
      x = x.squeeze()
      if target is None:
        classes = torch.norm(x, dim=2)
        max_length_indices = classes.max(dim=1)[1].squeeze()
      else:
        max_length_indices = target.max(dim=1)[1]
      masked = x.new_tensor(torch.eye(self.num_capsules))
      masked = masked.index_select(dim=0, index=max_length_indices.data)
      masked = masked.unsqueeze(-1)
      decoder_input = (x * masked).view(batch_size, -1)
      decoder_input = self.FC(decoder_input)
      decoder_input = decoder_input.view(batch_size,self.num_capsules, self.grid_size, self.grid_size)
      reconstructions = self.decoder(decoder_input)
      reconstructions = reconstructions.view(batch_size, self.im_channels, self.im_size, self.im_size)
      return reconstructions, masked

class GcnnSovnet(nn.Module):
  def __init__(self):
    super(GcnnSovnet,self).__init__()
    #(batch_size,1,33,33)
    #self.initial_conv = nn.Sequential(P4ConvZ2(in_channels=3, out_channels=64, kernel_size = 8),nn.SELU(),nn.BatchNorm3d(32))
    self.initial_conv = InitialP4Conv()
    #(batch_size,64,23,23)  
    self.primary_capsules = PrimaryCapsules(num_capsules=5,in_channels=32,out_channels=64)
    #(batch_size,10,32,4,21,21) 
    self.hidden_capsules1 = ConvolutionalCapsules(num_in_capsules=5,in_capsule_dim=64,num_out_capsules=5,out_capsule_dim=64,kernel_size=3)
    #(batch_size,10,32,4,19,19)
    self.hidden_capsules2 = ConvolutionalCapsules(num_in_capsules=5,in_capsule_dim=64,num_out_capsules=5,out_capsule_dim=64,kernel_size=5)
    #(batch_size,10,32,4,15,15)
    self.hidden_capsules3 = ConvolutionalCapsules(num_in_capsules=5,in_capsule_dim=64,num_out_capsules=5,out_capsule_dim=64,kernel_size=5)
    #(batch_size,10,32,4,11,11)
    self.hidden_capsules4 = ConvolutionalCapsules(num_in_capsules=5,in_capsule_dim=64,num_out_capsules=5,out_capsule_dim=64,kernel_size=5)
    #(batch_size,10,32,4,7,7)
    self.hidden_capsules5 = ConvolutionalCapsules(num_in_capsules=5,in_capsule_dim=64,num_out_capsules=5,out_capsule_dim=64,kernel_size=5)
    #(batch_size,10,32,4,3,3)
    self.class_capsules = ConvolutionalCapsules(num_in_capsules=5,in_capsule_dim=64,num_out_capsules=10,out_capsule_dim=64,kernel_size=3)
    #(batch_size,10,32,4,1,1)
    self.reconstruction_layer = ReconstructionLayer(10,64,33,3)
    #(batch_size,3,64,64)

  def forward(self,x,target):
      conv_output = self.initial_conv(x)
      primary_capsules = self.primary_capsules(conv_output)
      hidden_capsules_one, degree_score1 = self.hidden_capsules1(primary_capsules,routing_iterations=1)
      hidden_capsules_two, degree_score2 = self.hidden_capsules2(hidden_capsules_one,routing_iterations=2)
      hidden_capsules_three, degree_score3 = self.hidden_capsules3(hidden_capsules_two,routing_iterations=2)
      hidden_capsules_four, degree_score4 = self.hidden_capsules4(hidden_capsules_three,routing_iterations=1)
      hidden_capsules_five, degree_score5 = self.hidden_capsules5(hidden_capsules_four,routing_iterations=1)
      class_capsules, degree_scoreclass = self.class_capsules(hidden_capsules_five,routing_iterations=3)
      capsules =[hidden_capsules_one, hidden_capsules_two, hidden_capsules_three, hidden_capsules_four, hidden_capsules_five, class_capsules]
      degree_score = [degree_score1, degree_score2, degree_score3, degree_score4, degree_score5, degree_scoreclass]
      return capsules, degree_scorecheckpoint_file = ''#PLEASE FILL OUT WITH CHECKPOINT AFTER TRAINING CODE IN SAME FOLDER
check_point = torch.load(checkpoint_file)#,map_location='cpu')
model = GcnnSovnet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.DataParallel(model).to(device)
model.load_state_dict(check_point['model_state_dict'])
check_point = [] 
no_rotation = transforms.Compose([transforms.Resize(33),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
rotation = transforms.Compose([transforms.Resize(33),transforms.RandomRotation((180,180)),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
test_dataset1 = CIFAR10('./',train=False,download=True,transform=no_rotation)
test_dataset2 = CIFAR10('./',train=False,download=True,transform=rotation)
test_dataloader1 = DataLoader(test_dataset1,batch_size=128,shuffle=False)
test_dataloader2 = DataLoader(test_dataset2,batch_size=128,shuffle=False)
def onehot(tensor, num_classes=10):
    return torch.eye(num_classes).index_select(dim=0, index=tensor)# One-hot encode
def tensor_rotate(tensor,rotation):
    perm = [3,0,1,2]
    rotated_tensor = torch.flip(tensor,[5]).permute((0,1,2,3,5,4))[:,:,:,perm,:,:]
    if rotation == 90:
       times = 0
    if rotation == 180:
       times = 1
    if rotation == 270:
       times = 2
    if rotation not in (90,180,270):
        assert(False)
    while times != 0:
          rotated_tensor = torch.flip(rotated_tensor,[5]).permute((0,1,2,3,5,4))[:,:,:,perm,:,:]
          times -= 1
    return rotated_tensor
def rotate(tensor_list,rotation):
    rotated_tensor_list = []
    for tensor in tensor_list:
        rotated_tensor = tensor_rotate(tensor,rotation)
        rotated_tensor_list.append(rotated_tensor)
    return rotated_tensor_list
def mse(tensor_list1,tensor_list2):
    error = 0.0
    for tensor1, tensor2 in zip(tensor_list1,tensor_list2):
        error += F.mse_loss(tensor1,tensor2)
    return error
  capsules_error = 0.0
degree_score_error = 0.0
with torch.no_grad():
    model.eval()
    for batch_idx, ((data1, target1),(data2, target2)) in enumerate(zip(test_dataloader1,test_dataloader2)):
        target1 = onehot(target1).to(device)
        target2 = onehot(target2).to(device)
        capsules1, degree_score1 = model(data1,target1)
        capsules2, degree_score2 = model(data2,target2)
        rotated_capsules1 = rotate(capsules1,rotation=180)
        rotated_degree_score1 = rotate(degree_score1,rotation=180)
        capsules_error += mse(rotated_capsules1,capsules2)
        degree_score_error += mse(rotated_degree_score1,degree_score2)
print(capsules_error/float(len(test_dataset1)//128))
print(degree_score_error/float(len(test_dataset1)//128))
