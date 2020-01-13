import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim 
from torch.optim import lr_scheduler
import torch.nn.functional as F
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M, P4ConvZ2, P4ConvP4
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling

class BasicBlock(nn.Module):
    def __init__(self, in_planes=1, out_planes=32):
        super(BasicBlock, self).__init__()
        self.residue_conv1 = P4ConvP4(32, out_planes, kernel_size=3, stride=2, padding=1, bias=True)
        self.residue_bn1 = nn.BatchNorm3d(out_planes)
        self.residue_conv2 = P4ConvP4(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.residue_bn2 = nn.BatchNorm3d(out_planes)
        self.shortcut = nn.Sequential(
                P4ConvP4(32, out_planes, kernel_size=1, stride=2, bias=True),
                nn.BatchNorm3d(out_planes)
            )
        self.conv1 = P4ConvZ2(in_planes,32,kernel_size=3,stride=1,bias=True)

    def forward(self, x):
        #input: (batch_size, n_c, H, W)
        #output: (batch_size, out_channels, 4, H', W')   
        x = F.selu(self.conv1(x))
        out = F.selu(self.residue_bn1(self.residue_conv1(x)))
        out = self.residue_bn2(self.residue_conv2(out))
        out += self.shortcut(x)
        out = F.selu(out)
        return out      
     
class ResidualBlock(nn.Module):
  '''A residual block for use in capsule layers'''
  def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
    super(ResidualBlock,self).__init__()
    self.conv1 = P4ConvP4(in_channels,out_channels,kernel_size=3,stride=1,padding=1)#same convolution   
    self.bn1 = nn.BatchNorm3d(out_channels)
    self.conv2 = P4ConvP4(out_channels,out_channels,kernel_size,stride,padding)
    self.bn2 = nn.BatchNorm3d(out_channels)
    self.bridgeconv = P4ConvP4(in_channels,out_channels,kernel_size,stride,padding)
    self.bridgebn = nn.BatchNorm3d(out_channels)

  def forward(self,x):
    '''
        Input: (batch_size,in_channels,4,H,W)
        Output: (batch_size,out_channels,4,H',W')
    '''
    out = self.conv1(x) 
    out = F.selu(self.bn1(out))
    out = self.conv2(out)
    out = self.bn2(out)
    residue = self.bridgeconv(x)
    residue = self.bridgebn(residue)
    out = F.selu(out+residue)
    return out

class PrimaryCapsules(nn.Module):
  '''Use 2d convolution to extract capsules'''
  def __init__(self,num_capsules=10,in_channels=32,out_channels=16):
    super(PrimaryCapsules, self).__init__()
    self.num_capsules = num_capsules
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
    self.capsules = nn.ModuleList([
                                   nn.Sequential(
                                                 ResidualBlock(in_channels,out_channels,3,1,1),
                                                 ResidualBlock(out_channels,out_channels,3,1,1),
                                                 ResidualBlock(out_channels,out_channels,3,1,1)
                                   ) for i in range(num_capsules)])    
 
  def squash(self,x,dim):
    norm_squared = (x ** 2).sum(dim, keepdim=True)
    part1 = norm_squared / (1 +  norm_squared)
    part2 = x / torch.sqrt(norm_squared+ 1e-16)
    output = part1 * part2 
    return output
  
  def forward(self, x):
    #input: (batch_size, in_channels, H',W')
    #output: (batch_size, num_capsules, out_channels, H', W') 
    output = [caps(x) for caps in self.capsules]
    output = torch.stack(output, dim=1)
    return output
  
class PrimaryCapsules1(nn.Module):
  '''Use 2d convolution to extract capsules'''
  def __init__(self,num_capsules=10,in_channels=128,out_channels=16,kernel_size=3,stride=1,padding=0):
    super(PrimaryCapsules1, self).__init__()
    self.num_capsules = num_capsules
    self.capsules = nn.ModuleList([
          nn.Sequential(
                        ResidualBlock(in_channels,out_channels,kernel_size,stride,padding),
                        ResidualBlock(out_channels,out_channels,3,1,1)  
           )
           for i in range(num_capsules)
        ])
 
  def squash(self,x,dim):
    norm_squared = (x ** 2).sum(dim, keepdim=True)
    part1 = norm_squared / (1 +  norm_squared)
    part2 = x / torch.sqrt(norm_squared+ 1e-16)
    output = part1 * part2 
    return output
  
  def forward(self, x):
    #input: (batch_size, in_channels, H',W')
    #output: (batch_size, num_capsules, out_channels, H', W') 
    output = [caps(x) for caps in self.capsules]
    output = torch.stack(output, dim=1)
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
    self.projection_networks = nn.ModuleList([nn.Sequential(
                                ResidualBlock(in_capsule_dim,out_capsule_dim,kernel_size,stride,padding)) for i in range(num_out_capsules)
    ])

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
    grid_size = [capsules.size(-2),capsules.size(-1)]
    ### Prediction u_hat ####
    capsules = capsules.view(batch_size*self.num_in_capsules,self.in_capsule_dim,grid_size[0],grid_size[1])              
    u_hat = [p(capsules) for p in self.projection_networks]
    u_hat = torch.stack(u_hat,dim=1)
    grid_size = [u_hat.size(-2),u_hat.size(-1)]
    u_hat = u_hat.view(batch_size,self.num_in_capsules,self.num_out_capsules,self.out_capsule_dim,grid_size[0],grid_size[1])   
    ### u_hat:(batch_size,num_in_capsules,num_out_capsules,out_capsule_dim,H,W)
    u_hat_permute = u_hat.permute(0,2,4,5,1,3)#(batch_size,num_out_capsules,H,W,num_in_capsules,out_capsule_dim)
    affinity_matrices = self.cosine_similarity(u_hat_permute)    
    degree_score = F.softmax(torch.sum(affinity_matrices,dim=-1,keepdim=True),dim=4)#(batch_size,num_out_capsules,H,W,num_in_capsules,1)
    degree_score = (degree_score).permute(0,4,1,5,2,3)#(batch_size,num_in_capsules,num_out_capsules,1,H,W)
    s_j = (degree_score * u_hat).sum(dim=1)
    v_j = self.squash(s_j,dim=3)
    return s_j.squeeze(dim=1) 

  def forward(self,capsules,routing_iterations=1):
    '''
        Input: (batch_size, num_in_capsules, in_capsule_dim, H, W)
        Output: (batch_size, num_out_capsules, out_capsule_dim, H', W')
    '''
    out_capsules = self.degree_routing(capsules)
    return out_capsules
  
class ConvolutionalCapsules1(nn.Module):
  '''
      A capsule layer that uses one bottleneck layer per capsule-type
  '''
  def __init__(self,num_in_capsules,in_capsule_dim,num_out_capsules,out_capsule_dim,kernel_size,stride=1,padding=0):
    super(ConvolutionalCapsules1,self).__init__()
    self.num_in_capsules = num_in_capsules
    self.in_capsule_dim = in_capsule_dim
    self.num_out_capsules = num_out_capsules
    self.out_capsule_dim = out_capsule_dim
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.projection_networks = nn.ModuleList([nn.Sequential(
                                ResidualBlock(in_capsule_dim,out_capsule_dim,kernel_size,stride,padding)) for i in range(num_out_capsules)
    ])

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
    grid_size = [capsules.size(-2),capsules.size(-1)]
    ### Prediction u_hat ####
    capsules = capsules.view(batch_size*self.num_in_capsules,self.in_capsule_dim,grid_size[0],grid_size[1])              
    u_hat = [p(capsules) for p in self.projection_networks]
    u_hat = torch.stack(u_hat,dim=1)
    grid_size = [u_hat.size(-2),u_hat.size(-1)]
    u_hat = u_hat.view(batch_size,self.num_in_capsules,self.num_out_capsules,self.out_capsule_dim,grid_size[0],grid_size[1])   
    ### u_hat:(batch_size,num_in_capsules,num_out_capsules,out_capsule_dim,H,W)
    u_hat_permute = u_hat.permute(0,2,4,5,1,3)#(batch_size,num_out_capsules,H,W,num_in_capsules,out_capsule_dim)
    affinity_matrices = self.cosine_similarity(u_hat_permute)    
    degree_score = F.softmax(torch.sum(affinity_matrices,dim=-1,keepdim=True),dim=4)#(batch_size,num_out_capsules,H,W,num_in_capsules,1)
    degree_score = (degree_score).permute(0,4,1,5,2,3)#(batch_size,num_in_capsules,num_out_capsules,1,H,W)
    s_j = (degree_score * u_hat).sum(dim=1)
    v_j = self.squash(s_j,dim=3)
    return s_j.squeeze(dim=1) 

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
                                 nn.ConvTranspose2d(in_channels=self.num_capsules, out_channels=32, kernel_size=7, stride=2),
                                 nn.SELU(),
                                 nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=9, stride=1),
                                 nn.SELU(),
                                 nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=6, stride=1),
                                 nn.SELU(),
                                 nn.ConvTranspose2d(in_channels=32, out_channels=im_channels, kernel_size=5, stride=1),
                                 nn.SELU() 
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

class CnnSovnet(nn.Module):
  def __init__(self):
    super(CnnSovnet,self).__init__()
    self.initial_conv = BasicBlock().to(device)
    #(batch_size,32,13,13)
    self.primary_capsules = PrimaryCapsules(num_capsules=16,in_channels=32,out_channels=32).to(device).to(device)
    #(batch_size,10,16,13,13) 
    self.hidden_capsules_one = ConvolutionalCapsules(num_in_capsules=16,in_capsule_dim=32,num_out_capsules=16,out_capsule_dim=32,kernel_size=3).to(device)
    #(batch_size,10,16,11,11)
    self.hidden_capsules_two = ConvolutionalCapsules(num_in_capsules=16,in_capsule_dim=32,num_out_capsules=16,out_capsule_dim=32,kernel_size=3,stride=1,padding=1).to(device)
    #(batch_size,10,16,11,11)
    self.hidden_capsules_three = ConvolutionalCapsules(num_in_capsules=16,in_capsule_dim=32,num_out_capsules=16,out_capsule_dim=32,kernel_size=5).to(device)
    #(batch_size,10,16,7,7)
    self.hidden_capsules_four = ConvolutionalCapsules(num_in_capsules=16,in_capsule_dim=32,num_out_capsules=16,out_capsule_dim=32,kernel_size=3,stride=1,padding=1).to(device)
    #(batch_size,10,16,7,7)
    self.hidden_capsules_five = ConvolutionalCapsules(num_in_capsules=16,in_capsule_dim=32,num_out_capsules=16,out_capsule_dim=32,kernel_size=5).to(device)
    #(batch_size,10,16,3,3)
    self.class_capsules = ConvolutionalCapsules(num_in_capsules=16,in_capsule_dim=32,num_out_capsules=10,out_capsule_dim=32,kernel_size=3).to(device)
    #(batch_size,10,16,1,1)
    self.reconstruction_layer = ReconstructionLayer(10,32,28,1).to(device)
    #(batch_size,1,28,28)
        
  def forward(self,x,target):
      conv_output = self.initial_conv(x)
      primary_capsules = self.primary_capsules(conv_output)
      hidden_capsules_one = self.hidden_capsules_one(primary_capsules,routing_iterations=1)
      hidden_capsules_two = self.hidden_capsules_two(hidden_capsules_one,routing_iterations=2)
      hidden_capsules_three = self.hidden_capsules_three(hidden_capsules_two,routing_iterations=2)
      hidden_capsules_four = self.hidden_capsules_four(hidden_capsules_three,1)
      hidden_capsules_five = self.hidden_capsules_five(hidden_capsules_four,1)
      class_capsules = self.class_capsules(hidden_capsules_five,routing_iterations=3)
      reconstructions, masked = self.reconstruction_layer(class_capsules,target)
      return class_capsules, reconstructions, masked
  
  def get_activations(self,capsules):
    return torch.norm(capsules, dim=2).squeeze()
       
  def get_predictions(self,activations):
    max_length_indices = activations.max(dim=1)[1].squeeze()#(batch_size)
    predictions = activations.new_tensor(torch.eye(10))
    predictions = predictions.index_select(dim=0,index=max_length_indices)
    return predictions

  def loss(self,iteration,reconstructions,class_activations,target,data,lambda_=0.5,positive_margin=0.9, negative_margin=0.1):
    margin_loss = self.margin_loss(iteration,class_activations,target,lambda_,positive_margin,negative_margin)
    reconstruction_loss = self.sum_squared_loss(reconstructions,data)
    loss = margin_loss + 0.0005*reconstruction_loss
    return margin_loss, reconstruction_loss, loss
      
  def margin_loss(self,iteration,class_activations,target,lambda_=0.5,positive_margin=0.9, negative_margin=0.1):
    batch_size = class_activations.size(0)
    left = F.relu(positive_margin - class_activations).view(batch_size, -1) ** 2
    right = F.relu(class_activations - negative_margin).view(batch_size, -1) ** 2
    margin_loss = target * left + lambda_ *(1-target)*right
    margin_loss = margin_loss.sum(dim=1).mean()
    return margin_loss

  def sum_squared_loss(self,reconstructions,data):
      batch_size, img_channels, im_size, im_size = reconstructions.size()
      reconstruction_loss = F.mse_loss(data,reconstructions)
      return reconstruction_loss
    
class CnnSovnet1(nn.Module):
  def __init__(self):
    super(CnnSovnet1,self).__init__()
    self.initial_conv = ResidualBlock(1,128,3,1,1).to(device)
    #(batch_size,128,28,28)
    self.primary_capsules = PrimaryCapsules1(10,128,16,3,1,1).to(device)
    #(batch_size,10,16,28,28) 
    self.hidden_capsules_one = ConvolutionalCapsules1(num_in_capsules=10,in_capsule_dim=16,num_out_capsules=10,out_capsule_dim=16,kernel_size=3).to(device)
    #(batch_size,10,16,26,26)
    self.hidden_capsules_two = ConvolutionalCapsules1(num_in_capsules=10,in_capsule_dim=16,num_out_capsules=10,out_capsule_dim=16,kernel_size=3).to(device)
    #(batch_size,10,16,24,24)
    self.hidden_capsules_three = ConvolutionalCapsules1(num_in_capsules=10,in_capsule_dim=16,num_out_capsules=10,out_capsule_dim=16,kernel_size=5).to(device)
    #(batch_size,10,16,20,20)
    self.hidden_capsules_four = ConvolutionalCapsules1(num_in_capsules=10,in_capsule_dim=16,num_out_capsules=10,out_capsule_dim=16,kernel_size=3,stride=2).to(device)
    #(batch_size,10,16,9,9)
    self.hidden_capsules_five = ConvolutionalCapsules1(num_in_capsules=10,in_capsule_dim=16,num_out_capsules=10,out_capsule_dim=16,kernel_size=3,stride=2).to(device)
    #(batch_size,10,16,4,4)
    self.hidden_capsules_six = ConvolutionalCapsules1(num_in_capsules=10,in_capsule_dim=16,num_out_capsules=10,out_capsule_dim=16,kernel_size=3).to(device)
    #(batch_size,10,16,2,2)
    self.class_capsules = ConvolutionalCapsules1(num_in_capsules=10,in_capsule_dim=16,num_out_capsules=10,out_capsule_dim=16,kernel_size=2).to(device)
    #(batch_size,10,16,1,1)
    self.reconstruction_layer = ReconstructionLayer(10,16,28,1).to(device)
    #(batch_size,1,28,28)
        
  def forward(self,x,target):
      conv_output = self.initial_conv(x)
      primary_capsules = self.primary_capsules(conv_output,)
      hidden_capsules_one = self.hidden_capsules_one(primary_capsules,routing_iterations=1)
      hidden_capsules_two = self.hidden_capsules_two(hidden_capsules_one,routing_iterations=2)
      hidden_capsules_three = self.hidden_capsules_three(hidden_capsules_two,routing_iterations=2)
      hidden_capsules_four = self.hidden_capsules_four(hidden_capsules_three,routing_iterations=2)
      hidden_capsules_five = self.hidden_capsules_five(hidden_capsules_four,routing_iterations=2)
      hidden_capsules_six = self.hidden_capsules_six(hidden_capsules_five,routing_iterations=2)
      class_capsules = self.class_capsules(hidden_capsules_six,routing_iterations=3)
      reconstructions, masked = self.reconstruction_layer(class_capsules,target)
      return class_capsules, reconstructions, masked
  
  def get_activations(self,capsules):
    return torch.norm(capsules, dim=2).squeeze()
       
  def get_predictions(self,activations):
    max_length_indices = activations.max(dim=1)[1].squeeze()#(batch_size)
    predictions = activations.new_tensor(torch.eye(10))
    predictions = predictions.index_select(dim=0,index=max_length_indices)
    return predictions

  def loss(self,iteration,reconstructions,class_activations,target,data,lambda_=0.5,positive_margin=0.9, negative_margin=0.1):
    margin_loss = self.margin_loss(iteration,class_activations,target,lambda_,positive_margin, negative_margin)
    reconstruction_loss = self.sum_squared_loss(reconstructions,data)
    loss = margin_loss + 0.0005*reconstruction_loss
    return margin_loss, reconstruction_loss, loss
      
  def margin_loss(self,iteration,class_activations,target,lambda_=0.5,positive_margin=0.9, negative_margin=0.1):
    batch_size = class_activations.size(0)
    left = F.relu(positive_margin - class_activations).view(batch_size, -1) ** 2
    right = F.relu(class_activations - negative_margin).view(batch_size, -1) ** 2
    margin_loss = target * left + lambda_ *(1-target)*right
    margin_loss = margin_loss.sum(dim=1).mean()
    return margin_loss

  def sum_squared_loss(self,reconstructions,data):
      batch_size, img_channels, im_size, im_size = reconstructions.size()
      reconstruction_loss = F.mse_loss(data,reconstructions)
      return reconstruction_loss

'''Utility functions'''

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

def get_model(desc):
    if desc == 'CnnSovnet':
       return CnnSovnet().to(device)
    if desc == 'CnnSovnet1':
       return CnnSovnet1().to(device)      

def onehot(tensor, num_classes=10):
    return torch.eye(num_classes).index_select(dim=0, index=tensor)# One-hot encode

def get_loaders_mnist(translation_rotation_list):
    """Load MNIST dataset.
    The data is split and normalized between train and test sets.
    """
    train_loaders_desc = []
    test_loaders_desc = []
    for (translation,rotation) in translation_rotation_list:
        desc = str(translation[0])+'_'+str(translation[1])+'_'+str(rotation)+' '+'MNIST' 
        train_transform = transforms.Compose([
                                        transforms.RandomAffine(rotation,translation),  
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        test_transform = transforms.Compose([
                                             transforms.RandomAffine(rotation,translation),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])   
        training_set = MNIST('../../data/', train=True, download=True, transform=train_transform)
        training_data_loader = DataLoader(training_set, batch_size=64, shuffle=True)
        testing_set = MNIST('../../data/', train=False, download=True, transform=test_transform)
        testing_data_loader = DataLoader(testing_set, batch_size=100, shuffle=True)
        train_loaders_desc.append((training_data_loader,desc))
        test_loaders_desc.append((testing_data_loader,desc)) 
    return train_loaders_desc, test_loaders_desc
  
def check_point(model,optimizer,save_file,loss_history,epoch):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_history,
            }, save_file)

def test(model,test_loaders_desc,epoch,train_loader_desc,lambda_=0.5,positive_margin=0.9, negative_margin=0.1):
    test_accuracy_list = []
    test_loss_list = []
    model.eval()
    iteration = 0  
    for (test_loader,test_loader_desc) in test_loaders_desc:
        iteration += 1
        test_loss = 0.0
        margin_loss = 0.0
        reconstruction_loss = 0.0
        test_correct = 0.0
        iterations = 0
        for data, target in test_loader:
            iterations += 1
            target = onehot(target)
            data, target = data.to(device), target.to(device)
            capsules, reconstructions, _ = model(data,None)
            activations = model.get_activations(capsules)
            predictions = model.get_predictions(activations)
            margin_loss_temp, reconstruction_loss_temp, test_loss_temp = model.loss(iterations,reconstructions,activations,target,data,lambda_,positive_margin, negative_margin)# sum up batch loss
            margin_loss += margin_loss_temp
            reconstruction_loss += reconstruction_loss_temp
            test_loss += test_loss_temp
            test_correct += float((predictions.max(dim=1)[1] == target.max(dim=1)[1]).sum().item())
        test_loss /= len(test_loader.dataset)
        print(test_loader_desc) 
        print('\nTest set: Average loss: {:.4f}, Accuracy: {})\n'.format(
             test_loss, test_correct/len(test_loader.dataset),
             float(test_correct) / float(len(test_loader.dataset))))
        test_loss_list.append(test_loss.item())
        test_accuracy_list.append(test_correct/len(test_loader.dataset))
    return test_accuracy_list, test_loss_list

def train(baseline,num_epochs,train_loaders_desc,test_loaders_desc):
    for (train_loader, train_loader_desc) in train_loaders_desc:
        model = get_model(baseline)
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        model_name = baseline
        optimizer = optim.Adam(model.parameters())
        #optimizer = optim.SGD(model.parameters(),lr=1e-5,momentum=0.9)
        #cyclic_lr_scheduler = lr_scheduler.CyclicLR(optimizer,1e-5,1e-3)
        exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: max(1e-5,0.5**(epoch // 10)))
        lambda_=0.5
        positive_margin=0.9
        negative_margin=0.1
        save_file  = './check_points/'+model_name+'/'+train_loader_desc+'.pth'
        performance_file = './check_points/'+model_name+'/'+train_loader_desc+'.txt'
        loss_history = []
        f = open(performance_file, 'w')
        f.write("epoch, test_loss, train_loss_list, test_accuracy_list, train_acc\n")
        f.close() 
        for epoch in range(num_epochs):
            train_total = 0.0
            train_correct = 0.0
            train_loss_aggregate = 0.0
            model.train()
            iteration = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                iteration += 1
                target = onehot(target) 
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                capsules, reconstructions, _ = model(data,target)
                activations = model.get_activations(capsules)
                margin_loss, reconstruction_loss, loss = model.loss(iteration,reconstructions,activations,target,data,lambda_,positive_margin, negative_margin)
                train_loss_aggregate += loss.item() 
                loss_history.append(loss)
                loss.backward()
                #cyclic_lr_scheduler.step()
                #nn.utils.clip_grad_norm(model.parameters(), 1) 
                optimizer.step()
                predictions = model.get_predictions(activations)
                train_total += len(data)
                train_correct += (predictions.max(dim=1)[1] == target.max(dim=1)[1]).sum()
                train_accuracy = float(train_correct)/float(train_total)
                if batch_idx % 300 == 0:
                   print(model_name)
                   print(train_loader_desc)
                   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCorrect: {}'.format(
                         epoch, batch_idx * len(data), len(train_loader.dataset),
                         100. * batch_idx / len(train_loader), loss, train_accuracy))
                   print(margin_loss)
                   print(reconstruction_loss)
            with torch.no_grad():
                 test_accuracy_list, test_loss_list = test(model,test_loaders_desc,epoch,train_loader_desc)
                 print(test_accuracy_list)
            f = open(performance_file,'a')
            f.write(str(epoch)+ ' '+str(test_loss_list)+' '+str(train_loss_aggregate/len(train_loader.dataset))
                    +' '+str(test_accuracy_list)+' '+str(train_accuracy)+'\n')
            f.close()     
            if epoch%5 == 0:
               check_point(model,optimizer,save_file,loss_history,epoch)
            exp_lr_scheduler.step()
            

        #optimizer = optim.Adam(model.parameters())#restart with hard training
        #exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5**(epoch // 10))
        '''lambda_=0.8
        positive_margin=0.95
        negative_margin=0.05
        for epoch in range(num_epochs//2):
            train_total = 0.0
            train_correct = 0.0
            train_loss_aggregate = 0.0
            model.train()
            iteration = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                iteration += 1
                target = onehot(target) 
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                capsules, reconstructions, _ = model(data,target)
                activations = model.get_activations(capsules)
                margin_loss, reconstruction_loss, loss = model.loss(iteration,reconstructions,activations,target,data,lambda_,positive_margin, negative_margin)
                train_loss_aggregate += loss.item() 
                loss_history.append(loss)
                loss.backward()
                #nn.utils.clip_grad_norm(model.parameters(), 1) 
                optimizer.step()
                predictions = model.get_predictions(activations)
                train_total += len(data)
                train_correct += (predictions.max(dim=1)[1] == target.max(dim=1)[1]).sum()
                train_accuracy = float(train_correct)/float(train_total)
                if batch_idx % 100 == 0:
                   print(model_name)
                   print(train_loader_desc)
                   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCorrect: {}'.format(
                         epoch, batch_idx * len(data), len(train_loader.dataset),
                         100. * batch_idx / len(train_loader), loss, train_accuracy))
                   print(margin_loss)
                   print(reconstruction_loss)
            with torch.no_grad():
                 test_accuracy_list, test_loss_list = test(model,test_loaders_desc,epoch,train_loader_desc)
                 print(test_accuracy_list)
            f = open(performance_file,'a') 
            f.write(str(epoch)+ ' '+str(test_loss_list)+' '+str(train_loss_aggregate/len(train_loader.dataset))
                    +' '+str(test_accuracy_list)+' '+str(train_accuracy)+'\n')     
            f.close()
            if epoch%5 == 0:
               check_point(model,optimizer,save_file,loss_history,epoch)
            exp_lr_scheduler.step()'''        

def main_loop():
    baselines = ['CnnSovnet']
    num_epochs = 150
    translation_rotation_list = [((0.075,0.075),180),((0.075,0.075),90),((0,0),0)]
    train_loaders_desc, test_loaders_desc = get_loaders_mnist(translation_rotation_list)  
    for baseline in baselines:
        train(baseline,num_epochs,train_loaders_desc,test_loaders_desc)


main_loop()
