import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim 
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M, P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch.nn.functional as F

class InitialP4ResidualBlock(nn.Module):
   def __init__(self,in_channels,out_channels):
       super(InitialP4ResidualBlock,self).__init__()
       self.conv1 = P4ConvZ2(in_channels,out_channels,3,1,1)
       self.bn1 = nn.BatchNorm3d(out_channels)
       self.conv2 = P4ConvP4(out_channels,out_channels,3,1,1)
       self.bn2 = nn.BatchNorm3d(out_channels,out_channels)
       #self.shortcut = nn.Sequential()
       self.shortcut = nn.Sequential(P4ConvZ2(in_channels,out_channels,1),nn.BatchNorm3d(out_channels))

   def forward(self,x):
       out = F.selu(self.bn1(self.conv1(x)))
       out = self.bn2(self.conv2(out))
       out += self.shortcut(x)
       out = F.selu(out)
       return out     

class CapsuleP4ResidualBlock(nn.Module):
  '''Residual Block'''
  def __init__(self,num_out_capsules,in_capsule_dim,out_capsule_dim,stride=1):
      super(CapsuleP4ResidualBlock,self).__init__()
      self.conv1 = P4ConvP4(in_capsule_dim,out_capsule_dim,3,stride,1)
      self.bn1 = nn.BatchNorm3d(out_capsule_dim)
      self.conv2 = P4ConvP4(out_capsule_dim,num_out_capsules*out_capsule_dim,3,1,1)
      self.bn2 = nn.BatchNorm3d(num_out_capsules*out_capsule_dim)
      self.shortcut = nn.Sequential()
      self.shortcut = nn.Sequential(
                                       P4ConvP4(in_capsule_dim,num_out_capsules*out_capsule_dim, kernel_size=1, stride=stride),
                                       nn.BatchNorm3d(num_out_capsules*out_capsule_dim)
                                   )
  
  def forward(self,x):
      out = F.selu(self.bn1(self.conv1(x)))
      out = self.bn2(self.conv2(out))
      out += self.shortcut(x)
      out = F.selu(out)
      return out     

class PrimaryCapsules(nn.Module):
  '''Use 2d convolution to extract capsules'''
  def __init__(self,num_capsules=10,in_channels=32,out_channels=32):
    super(PrimaryCapsules, self).__init__()
    self.num_capsules = num_capsules
    self.out_channels = out_channels
    '''self.capsules = nn.ModuleList([
          nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.BatchNorm2d(out_channels),  
                        nn.SELU(),                         
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.BatchNorm2d(out_channels), 
                        nn.SELU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.BatchNorm2d(out_channels),    
                        nn.SELU(),
           )
           for i in range(num_capsules)
        ])'''
    #self.capsules = nn.ModuleList([
    #                               nn.Sequential(
    #                                             P4ResidualBlock(in_channels,out_channels,3,1,1),
    #                                             P4ResidualBlock(out_channels,out_channels,3,1,1),
    #                                             P4ResidualBlock(out_channels,out_channels,3,1,1)
    #                               ) for i in range(num_capsules)])
    #self.capsules = nn.Sequential(
    #                              P4ConvP4(in_channels,out_channels*num_capsules,kernel_size=5),
    #                              nn.BatchNorm3d(out_channels*num_capsules),
    #                             )
    self.capsules = nn.Sequential(
                                  P4ConvP4(in_channels,out_channels,3,1,1),
                                  nn.SELU(),
                                  nn.BatchNorm3d(out_channels),
                                  #P4ConvP4(out_channels,out_channels,3,1,1),
                                  #nn.SELU(),
                                  #nn.BatchNorm3d(out_channels),  
                                  CapsuleP4ResidualBlock(num_capsules,out_channels,out_channels,stride=1)
                                 )   
       
  def squash(self,x,dim):
    norm_squared = (x ** 2).sum(dim, keepdim=True)
    part1 = norm_squared / (1 +  norm_squared)
    part2 = x / torch.sqrt(norm_squared+ 1e-16)
    output = part1 * part2 
    return output
  
  def forward(self, x):
    #input: (batch_size, in_channels, 4, H',W')
    #output: (batch_size, num_capsules, out_channels, 4, H', W') 
    #output = [caps(x) for caps in self.capsules]
    #output = torch.stack(output, dim=1)
    output = self.capsules(x)
    H, W = output.size(-2), output.size(-1)
    output = output.view(-1,self.num_capsules,self.out_channels,4,H,W)
    output = self.squash(output,dim=2)    
    return output

class ConvolutionalCapsules(nn.Module):
  '''
      A capsule layer that uses one bottleneck layer per capsule-type
  '''
  def __init__(self,num_in_capsules,in_capsule_dim,num_out_capsules,out_capsule_dim,kernel_size=7,stride=1,padding=0,class_=False):
    super(ConvolutionalCapsules,self).__init__()
    self.num_in_capsules = num_in_capsules
    self.in_capsule_dim = in_capsule_dim
    self.num_out_capsules = num_out_capsules
    self.out_capsule_dim = out_capsule_dim
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    #self.projection_networks = nn.ModuleList([nn.Sequential(
    #                            P4ResidualBlock(in_capsule_dim,out_capsule_dim,kernel_size,stride,padding)) for i in range(num_out_capsules)
    #])
    #self.projection_network = nn.Sequential(P4ResidualBlock(in_capsule_dim,out_capsule_dim*num_out_capsules,kernel_size,stride,padding))
    #self.projection_network = nn.Sequential(P4ConvP4(in_capsule_dim,out_capsule_dim*num_out_capsules,kernel_size,stride,padding),
    #                                        nn.BatchNorm3d(out_capsule_dim*num_out_capsules))
    if class_ == False:
       self.projection_network = CapsuleP4ResidualBlock(num_out_capsules,in_capsule_dim,out_capsule_dim,stride)
    else:
       self.projection_network = nn.Sequential(P4ConvP4(in_capsule_dim,out_capsule_dim*num_out_capsules,kernel_size,stride,padding))  
     
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
    #u_hat = [p(capsules) for p in self.projection_networks]
    #u_hat = torch.stack(u_hat,dim=1)    
    u_hat = self.projection_network(capsules)
    grid_size = [u_hat.size(-3),u_hat.size(-2),u_hat.size(-1)]
    u_hat = u_hat.view(batch_size,self.num_in_capsules,self.num_out_capsules,self.out_capsule_dim,grid_size[0],grid_size[1],grid_size[2])   
    ### u_hat:(batch_size,num_in_capsules,num_out_capsules,out_capsule_dim,4,H,W)
    u_hat_permute = u_hat.permute(0,2,4,5,6,1,3)#(batch_size,num_out_capsules,4,H,W,num_in_capsules,out_capsule_dim)
    affinity_matrices = self.cosine_similarity(u_hat_permute)    
    degree_score = F.softmax(torch.sum(affinity_matrices,dim=-1,keepdim=True),dim=5)#(batch_size,num_out_capsules,4,H,W,num_in_capsules,1)
    degree_score = (degree_score).permute(0,5,1,6,2,3,4)#(batch_size,num_in_capsules,num_out_capsules,1,4,H,W)
    s_j = (degree_score * u_hat).sum(dim=1)
    v_j = self.squash(s_j,dim=3)
    return v_j.squeeze(dim=1) 

  def forward(self,capsules,routing_iterations=1):
    '''
        Input: (batch_size, num_in_capsules, in_capsule_dim, H, W)
        Output: (batch_size, num_out_capsules, out_capsule_dim, H', W')
    '''
    out_capsules = self.degree_routing(capsules)
    return out_capsules

class P4PoolingCapsules(nn.Module):
  '''
      A capsule layer that pools capsules of the same type
  '''
  def __init__(self,num_capsules,capsule_dim,kernel_size,stride=1):
    super(P4PoolingCapsules,self).__init__()
    self.num_capsules = num_capsules
    self.capsule_dim = capsule_dim
    self.kernel_size = kernel_size
    self.stride = stride   

  def squash(self,x,dim):
    norm_squared = (x ** 2).sum(dim, keepdim=True)
    part1 = norm_squared / (1 +  norm_squared)
    part2 = x / torch.sqrt(norm_squared+ 1e-16)
    output = part1 * part2 
    return output
   
  def forward(self,capsules):
    '''
        Input: (batch_size, num_capsules, capsule_dim, 4, H, W)
        Output: (batch_size, num_capsules, capsule_dim, 4, H', W')
    '''
    batch_size = capsules.size(0)
    H = capsules.size(-1)
    capsules = capsules.view(batch_size*self.num_capsules,self.capsule_dim,4,H,H)
    out_capsules = plane_group_spatial_max_pooling(capsules,self.kernel_size,self.stride)
    H = out_capsules.size(-1)
    out_capsules = out_capsules.view(batch_size,self.num_capsules,self.capsule_dim,4,H,H)
    return self.squash(out_capsules,dim=2)   
  
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
                                 nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=1),
                                 #nn.SELU(),
                                 #nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=8, stride=2),  
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
    #(batch_size,1,28,28)
    #self.initial_conv = nn.Sequential(P4ConvZ2(in_channels=3, out_channels=64, kernel_size = 8),nn.SELU(),nn.BatchNorm3d(32))
    self.initial_conv = InitialP4ResidualBlock(1,64)
    #(batch_size,64,28,28)  
    self.primary_capsules = PrimaryCapsules(num_capsules=5,in_channels=64,out_channels=64)
    #(batch_size,10,32,4,28,28) 
    #self.hidden_capsules1 = ConvolutionalCapsules(num_in_capsules=8,in_capsule_dim=32,num_out_capsules=8,out_capsule_dim=32,stride=1)
    #(batch_size,10,32,4,28,28)
    self.hidden_capsules1 = ConvolutionalCapsules(num_in_capsules=5,in_capsule_dim=64,num_out_capsules=5,out_capsule_dim=64,stride=2)
    #(batch_size,10,32,4,14,14)
    self.hidden_capsules2 = ConvolutionalCapsules(num_in_capsules=5,in_capsule_dim=64,num_out_capsules=5,out_capsule_dim=64,stride=1)
    #(batch_size,10,32,4,14,14)
    #self.hidden_capsules4 = ConvolutionalCapsules(num_in_capsules=8,in_capsule_dim=32,num_out_capsules=8,out_capsule_dim=32,stride=1)
    #(batch_size,10,32,4,14,14)
    self.hidden_capsules3 = ConvolutionalCapsules(num_in_capsules=5,in_capsule_dim=64,num_out_capsules=5,out_capsule_dim=64,stride=2)
    #(batch_size,10,32,4,7,7)
    self.hidden_capsules4 = ConvolutionalCapsules(num_in_capsules=5,in_capsule_dim=64,num_out_capsules=5,out_capsule_dim=64,stride=1)
    #(batch_size,10,32,4,7,7)
    self.hidden_capsules5 = ConvolutionalCapsules(num_in_capsules=5,in_capsule_dim=64,num_out_capsules=5,out_capsule_dim=64,stride=3)
    #(batch_size,10,32,4,3,3)
    #self.hidden_capsules6 = ConvolutionalCapsules(num_in_capsules=5,in_capsule_dim=64,num_out_capsules=5,out_capsule_dim=64,stride=1)
    #(batch_size,10,32,4,3,3)  
    self.class_capsules = ConvolutionalCapsules(num_in_capsules=5,in_capsule_dim=64,num_out_capsules=10,out_capsule_dim=64,kernel_size=3,class_=True)
    #(batch_size,10,32,4,1,1)
    self.reconstruction_layer = ReconstructionLayer(10,64,28,1)
    #(batch_size,3,64,64)

  def forward(self,x,target):
      conv_output = self.initial_conv(x)

      primary_capsules = self.primary_capsules(conv_output)
      hidden_capsules_one = self.hidden_capsules1(primary_capsules,routing_iterations=1)
      hidden_capsules_two = self.hidden_capsules2(hidden_capsules_one,routing_iterations=2)
      hidden_capsules_three = self.hidden_capsules3(hidden_capsules_two,routing_iterations=2)
      hidden_capsules_four = self.hidden_capsules4(hidden_capsules_three,routing_iterations=1)
      hidden_capsules_five = self.hidden_capsules5(hidden_capsules_four,routing_iterations=1)
      #hidden_capsules_six =  self.hidden_capsules6(hidden_capsules_five,routing_iterations=1)
      #hidden_capsules_seven =  self.hidden_capsules7(hidden_capsules_six,routing_iterations=1)
      #hidden_capsules_eight =  self.hidden_capsules8(hidden_capsules_seven,routing_iterations=1)
      class_capsules = self.class_capsules(hidden_capsules_five,routing_iterations=3)
      #print(class_capsules.size())
      class_capsules = class_capsules.squeeze()
      class_capsules_norm = torch.norm(class_capsules,dim=2,keepdim=False)
      max_length_indices_per_type = torch.max(class_capsules_norm,dim=2)[1]  
      masked = class_capsules.new_tensor(torch.eye(class_capsules.size(3)))
      masked = masked.index_select(dim=0, index=max_length_indices_per_type.data.view(-1))
      masked = masked.view(class_capsules.size(0),class_capsules.size(1),-1).unsqueeze(2)
      class_capsules_output = (class_capsules*masked).sum(3)
      class_capsules_output = class_capsules_output.unsqueeze(-1).unsqueeze(-1)
      reconstructions, masked = self.reconstruction_layer(class_capsules_output,target)
      return class_capsules_output, reconstructions, masked
  
def get_activations(capsules):
    return torch.norm(capsules, dim=2).squeeze()
       
def get_predictions(activations):
     max_length_indices = activations.max(dim=1)[1].squeeze()#(batch_size)
     predictions = activations.new_tensor(torch.eye(10))
     predictions = predictions.index_select(dim=0,index=max_length_indices)
     return predictions

def loss_(iteration,reconstructions,class_activations,target,data,lambda_=0.5,positive_margin=0.9, negative_margin=0.1):
     margin_loss = margin_loss_(iteration,class_activations,target,lambda_,positive_margin,negative_margin)
     reconstruction_loss = sum_squared_loss(reconstructions,data)
     loss = margin_loss + 0.0005*reconstruction_loss
     return margin_loss, reconstruction_loss, loss
      
def margin_loss_(iteration,class_activations,target,lambda_=0.5,positive_margin=0.9, negative_margin=0.1):
    batch_size = class_activations.size(0)
    left = F.relu(positive_margin - class_activations).view(batch_size, -1) ** 2
    right = F.relu(class_activations - negative_margin).view(batch_size, -1) ** 2
    margin_loss = target * left + lambda_ *(1-target)*right
    margin_loss = margin_loss.sum(dim=1).mean()
    return margin_loss

def sum_squared_loss(reconstructions,data):
    batch_size, img_channels, im_size, im_size = reconstructions.size()
    reconstruction_loss = F.mse_loss(data,reconstructions)
    return reconstruction_loss    

'''class GcnnSovnet(nn.Module):
  def __init__(self):
    super(GcnnSovnet,self).__init__()
    self.initial_conv = P4BasicBlock()
    #(batch_size,32,4,15,15)
    self.primary_capsules = PrimaryCapsules(num_capsules=10,in_channels=32,out_channels=32)
    #(batch_size,8,16,4,15,15) 
    self.hidden_capsules_one = ConvolutionalCapsules(num_in_capsules=10,in_capsule_dim=32,num_out_capsules=10,out_capsule_dim=32,kernel_size=3)
    #(batch_size,8,16,4,13,13)
    self.hidden_capsules_two = ConvolutionalCapsules(num_in_capsules=10,in_capsule_dim=32,num_out_capsules=10,out_capsule_dim=32,kernel_size=3)
    #(batch_size,8,16,4,11,11)
    self.hidden_capsules_three = ConvolutionalCapsules(num_in_capsules=10,in_capsule_dim=32,num_out_capsules=10,out_capsule_dim=32,kernel_size=3,stride=1,padding=1)
    #(batch_size,8,16,4,11,11)
    self.hidden_capsules_four = ConvolutionalCapsules(num_in_capsules=10,in_capsule_dim=32,num_out_capsules=10,out_capsule_dim=32,kernel_size=5)
    #(batch_size,8,16,4,7,7)
    self.hidden_capsules_five = ConvolutionalCapsules(num_in_capsules=10,in_capsule_dim=32,num_out_capsules=10,out_capsule_dim=32,kernel_size=3,stride=1,padding=1)
    #(batch_size,8,16,4,7,7)
    self.hidden_capsules_six = ConvolutionalCapsules(num_in_capsules=10,in_capsule_dim=32,num_out_capsules=32,out_capsule_dim=10,kernel_size=5)
    #(batch_size,8,16,4,3,3)
    self.class_capsules = ConvolutionalCapsules(num_in_capsules=10,in_capsule_dim=32,num_out_capsules=10,out_capsule_dim=32,kernel_size=3)
    #(batch_size,10,16,4,1,1)
    self.reconstruction_layer = ReconstructionLayer(10,32,32,3)
    #(batch_size,1,28,28)
        
  def forward(self,x,target):
      conv_output = self.initial_conv(x)
      primary_capsules = self.primary_capsules(conv_output)
      hidden_capsules_one = self.hidden_capsules_one(primary_capsules,routing_iterations=1)
      hidden_capsules_two = self.hidden_capsules_two(hidden_capsules_one,routing_iterations=2)
      hidden_capsules_three = self.hidden_capsules_three(hidden_capsules_two,routing_iterations=2)
      hidden_capsules_four = self.hidden_capsules_four(hidden_capsules_three,1)
      hidden_capsules_five = self.hidden_capsules_five(hidden_capsules_four,1)
      hidden_capsules_six = self.hidden_capsules_six(hidden_capsules_five,1)
      class_capsules = self.class_capsules(hidden_capsules_six,routing_iterations=3)
      class_capsules = class_capsules.squeeze()
      class_capsules_norm = torch.norm(class_capsules,dim=2,keepdim=False)
      max_length_indices_per_type = torch.max(class_capsules_norm,dim=2)[1]  
      masked = class_capsules.new_tensor(torch.eye(class_capsules.size(3)))
      masked = masked.index_select(dim=0, index=max_length_indices_per_type.data.view(-1))
      masked = masked.view(class_capsules.size(0),class_capsules.size(1),-1).unsqueeze(2)
      class_capsules_output = (class_capsules*masked).sum(3)
      class_capsules_output = class_capsules_output.unsqueeze(-1).unsqueeze(-1)  
      reconstructions, masked = self.reconstruction_layer(class_capsules_output,target)
      return class_capsules_output, reconstructions, masked
  
def get_activations(capsules):
    return torch.norm(capsules, dim=2).squeeze()
       
def get_predictions(activations):
    max_length_indices = activations.max(dim=1)[1].squeeze()#(batch_size)
    predictions = activations.new_tensor(torch.eye(10))
    predictions = predictions.index_select(dim=0,index=max_length_indices)
    return predictions

def loss_(iteration,reconstructions,class_activations,target,data,lambda_=0.5,positive_margin=0.9, negative_margin=0.1):
    margin_loss = margin_loss_(iteration,class_activations,target,lambda_,positive_margin,negative_margin)
    reconstruction_loss = sum_squared_loss(reconstructions,data)
    loss = margin_loss + 0.0005*reconstruction_loss
    return margin_loss, reconstruction_loss, loss
      
def margin_loss_(iteration,class_activations,target,lambda_=0.5,positive_margin=0.9, negative_margin=0.1):
    batch_size = class_activations.size(0)
    left = F.relu(positive_margin - class_activations).view(batch_size, -1) ** 2
    right = F.relu(class_activations - negative_margin).view(batch_size, -1) ** 2
    margin_loss = target * left + lambda_ *(1-target)*right
    margin_loss = margin_loss.sum(dim=1).mean()
    return margin_loss

def sum_squared_loss(reconstructions,data):
      batch_size, img_channels, im_size, im_size = reconstructions.size()
      reconstruction_loss = F.mse_loss(data,reconstructions)
      return reconstruction_loss'''
