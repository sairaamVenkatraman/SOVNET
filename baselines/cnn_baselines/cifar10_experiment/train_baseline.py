"""
   This file trains CNN and Group equivariant CNN baselines on CIFAR10 for three types of affine transformations at train and test level  
"""
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from cnn_resnet_baselines import *
from gcnn_conv_baselines import *
from torch import optim
from torch.optim import SGD
from torch.optim import lr_scheduler   
import torch.nn.functional as F

DATAPATH = "../../../data/CIFAR10/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
SAVEPATH = "./logfiles/CIFAR10/"
BATCHSIZE = 128
INDEX_OF_TRAIN_TRIALS = 1
NUMBER_OF_TEST_TRIALS = 3

def get_model(desc):#function to load a model. Used to automate running through all the conv baselines
    if desc == "SimpleConvP4":
       return SimpleConvP4().to(DEVICE)
    if desc == "SimpleConvP4M":
       return SimpleConvP4M().to(DEVICE)  
    if desc == "CnnResNet18":
       return CnnResNet18().to(DEVICE)
    if desc == "CnnResNet34":
       return CnnResNet34().to(DEVICE)
    if desc == "CnnResNet50":
       return CnnResNet50().to(DEVICE)

def main_loop():
    baseline_list = ["SimpleConvP4"]#list of conv baselines
    #list of affine transformations. translations and rotations
    train_translation_rotation_list = [((0,0),0),((0.066,0.066),30),((0.066,0.066),60),((0.066,0.066),90),((0.066,0.066),180)]
    test_translation_rotation_list = [((0,0),0),((0.066,0.066),30),((0.066,0.066),60),((0.066,0.066),90),((0.066,0.066),180)]
    train_loaders_desc, test_loaders_desc = get_loaders_cifar10(train_translation_rotation_list,test_translation_rotation_list,BATCHSIZE)  
    for baseline in baseline_list:
        train_test(baseline,train_loaders_desc,test_loaders_desc) 
 
def get_loaders_cifar10(train_translation_rotation_list,test_translation_rotation_list,batch_size):
    """Load cifar10 dataset. The data is divided by 255 and subracted by mean and divided by standard deviation.
    """
    train_loaders_desc = []
    test_loaders_desc = []
    for (translation,rotation) in train_translation_rotation_list:
        train_desc = 'train_'+str(translation[0])+'_'+str(translation[1])+'_'+str(rotation)+'_'+'CIFAR10'#for identifying in logs 
        train_transform = transforms.Compose([
                                        transforms.RandomAffine(rotation,translation),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ])
        training_set = CIFAR10(DATAPATH, train=True, download=True, transform=train_transform)
        training_data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        train_loaders_desc.append((training_data_loader,train_desc))
    for (translation,rotation) in test_translation_rotation_list:
        test_desc = 'test_'+str(translation[0])+'_'+str(translation[1])+'_'+str(rotation)+'_'+'CIFAR10'#for identifying in logs  
        test_transform = transforms.Compose([
                                        transforms.RandomAffine(rotation,translation),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ])
        testing_set = CIFAR10(DATAPATH, train=False, download=True, transform=test_transform)  
        testing_data_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True)
        test_loaders_desc.append((testing_data_loader,test_desc)) 
    return train_loaders_desc, test_loaders_desc

def train_test(baseline,train_loaders_desc,test_loaders_desc):
    for (train_loader, train_loader_desc) in train_loaders_desc:
        model = get_model(baseline)
        if baseline == "CnnResNet18":
           criterion = nn.CrossEntropyLoss()
           optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
           scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
           num_epochs = 350
        if baseline == "CnnResNet34":
           criterion = nn.CrossEntropyLoss()
           optimizer = optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
           scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[150,250],gamma=0.1)
           num_epochs = 350
        if baseline == "SimpleConvP4":
           criterion = nn.CrossEntropyLoss()
           optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
           scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
           num_epochs = 200
        if baseline == "SimpleConvP4M":
           criterion = nn.CrossEntropyLoss()
           optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
           scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 250, 300], gamma=0.1)
           num_epochs = 350
        if baseline == "CnnResNet50":
           criterion = nn.CrossEntropyLoss()
           optimizer = optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
           scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[150,250],gamma=0.1)
           num_epochs = 350     
        model_name = baseline
        checkpoint_file  = SAVEPATH+model_name+'/'+train_loader_desc+'train_trail_'+str(INDEX_OF_TRAIN_TRIALS)+'.pth'
        performance_file  = SAVEPATH+model_name+'/'+train_loader_desc+'train_trail_'+str(INDEX_OF_TRAIN_TRIALS)+'.txt'
        f = open(performance_file, 'w')#create the performance logfile
        f.write("epoch,\ttest_loss,\ttrain_loss_list,\ttest_accuracy_list,\ttrain_acc\n")
        f.close()
        loss_history = []
        for epoch in range(num_epochs):
            train_total = 0.0#for running accuracy
            train_correct = 0.0#for running accuracy
            train_loss_aggregate = 0.0#aggregate train loss
            model.train()#set model to train
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                train_loss_aggregate += loss.item()
                loss_history.append(loss) 
                loss.backward()
                optimizer.step()
                pred = output.max(1, keepdim=True)[1]#obtain predictions
                train_total += len(data)
                train_correct += pred.eq(target.data.view_as(pred)).cpu().sum()#obtain train accuracy
                train_accuracy = float(train_correct)/float(train_total)
                if batch_idx % 100 == 0:
                   print(model_name)
                   print(train_loader_desc)
                   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCorrect: {}'.format(
                         epoch, batch_idx * len(data), len(train_loader.dataset),
                         100. * batch_idx / len(train_loader), loss, train_accuracy))
            with torch.no_grad():
                 test_accuracy_list, test_loss_list = test(model,test_loaders_desc,epoch,train_loader_desc)
            f = open(performance_file, 'a')
            f.write(str(epoch)+ ' '+str(test_loss_list)+' '+str(train_loss_aggregate/len(train_loader.dataset))
                    +' '+str(test_accuracy_list)+' '+str(train_accuracy)+'\n')
            f.close()
            scheduler.step()     
            if epoch%1 == 0:
               check_point(model,optimizer,checkpoint_file+'_'+str(epoch),loss_history,epoch)
        torch.cuda.empty_cache() 

def test(model,test_loaders_desc,epoch,train_loader_desc):
    test_accuracy_list = []
    test_loss_list = []
    model.eval()
    for (test_loader,test_loader_desc) in test_loaders_desc:
        test_loss = 0.0
        test_correct = 0.0
        for trial in range(NUMBER_OF_TEST_TRIALS):
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                test_loss += F.cross_entropy(output, target, size_average=False) # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                test_correct += float(pred.eq(target.data.view_as(pred)).cpu().sum())
        test_loss /= float(len(test_loader.dataset)*NUMBER_OF_TEST_TRIALS)
        test_correct /= float(len(test_loader.dataset)*NUMBER_OF_TEST_TRIALS)  
        print(test_loader_desc) 
        print('\nTest set: Average loss: {:.4f}, Accuracy: {})\n'.format(test_loss, test_correct))
        test_loss_list.append(test_loss.item())
        test_accuracy_list.append(test_correct)
    return test_accuracy_list, test_loss_list

def check_point(model,optimizer,save_file,loss_history,epoch):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_history,
            }, save_file)

main_loop()            
