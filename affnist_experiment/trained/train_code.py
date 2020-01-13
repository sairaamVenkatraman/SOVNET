import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim 
from tqdm import tqdm
from torch.optim import lr_scheduler
import torch.nn.functional as F
import sys
sys.path.append('./')

from dataset_loader import *
from data_parallel_model import *


DATAPATH = "../../data/"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
SAVEPATH = "./check_points/MNIST/"
BATCHSIZE = 64
INDEX_OF_TRAIN_TRIALS = 1
NUMBER_OF_TEST_TRIALS = 1

def get_model(desc):
    if desc == 'GcnnSovnet':
       model = GcnnSovnet()
       model = nn.DataParallel(model).to(DEVICE)
       return model      

def onehot(tensor, num_classes=10):
    return torch.eye(num_classes).index_select(dim=0, index=tensor)# One-hot encode

def check_point(model,optimizer,save_file,loss_history,epoch):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_history,
            }, save_file)

def train_test(baseline,num_epochs,train_loaders_desc,test_loaders_desc):
    for (train_loader, train_loader_desc) in train_loaders_desc:
        train_loader1 = train_loader
        train_loader2 = train_loader
        train_loader1 = train_loader1
        train_loader2 = train_loader2 
        model = get_model(baseline)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(pytorch_total_params)
        model_name = baseline
        optimizer = optim.Adam(model.parameters(),lr=0.001)
        #exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5**(epoch // 10))
        #step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer,milestones) 
        #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer,lambda epoch: max(1e-3,0.96**epoch))#lambda epoch: 0.5**(epoch // 10))
        #exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,150, 250], gamma=0.1)
        save_file  = SAVEPATH+model_name+'/'+train_loader_desc+'train_trial_'+str(INDEX_OF_TRAIN_TRIALS)+'.pth'
        performance_file = SAVEPATH+model_name+'/'+train_loader_desc+'train_trial_'+str(INDEX_OF_TRAIN_TRIALS)+'.txt'
        f = open(performance_file, 'w')
        f.write("epoch, test_loss, train_loss_list, test_accuracy_list, train_acc\n")
        f.close()
        train(model,optimizer,exp_lr_scheduler,train_loader1,test_loaders_desc,save_file,performance_file,num_epochs//2,0.5,0.9,0.1,model_name,train_loader_desc)
        optimizer = optim.Adam(model.parameters(),lr=2e-4)
        #milestones = [10,30,40]
        #step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer,milestones)
        #exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5**(epoch // 10))
        exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: max(1e-3,0.96**epoch))#lr_lambda=lambda epoch: 0.5**(epoch // 10))
        train(model,optimizer,exp_lr_scheduler,train_loader2,test_loaders_desc,save_file,performance_file,num_epochs//2,0.8,0.95,0.05,model_name,train_loader_desc)
         
def train(model,optimizer,exp_lr_scheduler,train_loader,test_loaders_desc,save_file,performance_file,num_epochs,lambda_,positive_margin,negative_margin,train_loader_desc,model_name):
    for epoch in range(num_epochs):
        print("epoch ",epoch)
        train_total = 0.0
        train_correct = 0.0
        train_loss_aggregate = 0.0
        loss_history = []
        model.train()
        iteration = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            iteration += 1
            target = onehot(target) 
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            capsules, reconstructions, _ = model(data,target)
            activations = get_activations(capsules)
            margin_loss, reconstruction_loss, loss = loss_(iteration,reconstructions,activations,target,data,lambda_,positive_margin, negative_margin)
            train_loss_aggregate += loss.item() 
            loss.backward()
            optimizer.step()
            predictions = get_predictions(activations)
            train_total += len(data)
            train_correct += (predictions.max(dim=1)[1] == target.max(dim=1)[1]).sum()
            train_accuracy = float(train_correct)/float(train_total)
            '''if batch_idx % 10 == 0:
               print(model_name)
               print(train_loader_desc)
               #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tCorrect: {}'.format(
               #      epoch, batch_idx * len(data), len(train_loader.dataset),
               #      100. * batch_idx / len(train_loader), loss, train_accuracy))
               print('loss {}'.format(loss))
               print('margin loss {}'.format(margin_loss))
               print('reconstruction loss {}'.format(reconstruction_loss))
               print('train_accuracy {}'.format(train_accuracy))'''
        loss_history.append(train_loss_aggregate/len(train_loader))
        print('train_accuracy {}'.format(train_accuracy))
        #print('testing')
        if epoch == 74 or epoch == 149 or epoch%30 == 0:
           with torch.no_grad():
               test_accuracy_list, test_loss_list = test(model,test_loaders_desc,epoch,train_loader_desc)
               print(test_accuracy_list)
           f = open(performance_file,'a')
           f.write(str(epoch)+ ' '+str(test_loss_list)+' '+str(train_loss_aggregate/len(train_loader))
                +' '+str(test_accuracy_list)+' '+str(train_accuracy)+'\n')
           f.close()     
        if epoch%10 == 0:
           check_point(model,optimizer,save_file+'_'+str(epoch),loss_history,epoch)
        #with torch.no_grad():
        #       test_accuracy_list, test_loss_list = test(model,test_loaders_desc,epoch,train_loader_desc)
        #       print(test_accuracy_list)
        #f = open(performance_file,'a')
        #f.write(str(epoch)+ ' '+str(test_loss_list)+' '+str(train_loss_aggregate/len(train_loader))
        #        +' '+str(test_accuracy_list)+' '+str(train_accuracy)+'\n')
        #f.close()
        exp_lr_scheduler.step()

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
         test_total = 0.0 
         iterations = 0
         for trial in range(NUMBER_OF_TEST_TRIALS):
             for data, target in test_loader:
                 iterations += 1
                 target = onehot(target)
                 data, target = data.to(DEVICE), target.to(DEVICE)
                 capsules, reconstructions, _ = model(data,None)
                 activations = get_activations(capsules)
                 predictions = get_predictions(activations)
                 margin_loss_temp, reconstruction_loss_temp, test_loss_temp = loss_(iterations,reconstructions,activations,target,data,lambda_,positive_margin, negative_margin)# sum up batch loss
                 margin_loss += margin_loss_temp
                 reconstruction_loss += reconstruction_loss_temp
                 test_loss += test_loss_temp
                 test_correct += float((predictions.max(dim=1)[1] == target.max(dim=1)[1]).sum().item())
                 test_total += len(data)
         test_loss /= float(test_total)
         test_correct /= float(test_total)
         print(test_loader_desc) 
         print('\nTest set: Average loss: {:.4f}, Accuracy: {})\n'.format(test_loss, test_correct))
         test_loss_list.append(test_loss.item())
         test_accuracy_list.append(test_correct)
    return test_accuracy_list, test_loss_list

def main_loop():
    baselines = ['GcnnSovnet']
    num_epochs = 150
    train_translation_rotation_list = [((0.15,0.15),30)]#,((0.075,0.075),30),((0.075,0.075),60),((0.075,0.075),90),((0.075,0.075),180)]
    test_translation_rotation_list = [((0,0),0)]#,((0.075,0.075),30),((0.075,0.075),60),((0.075,0.075),90),((0.075,0.075),180)]
    train_loaders_desc, test_loaders_desc = get_loaders_mnist(train_translation_rotation_list,test_translation_rotation_list,BATCHSIZE)  
    for baseline in baselines:
        train_test(baseline,num_epochs,train_loaders_desc,test_loaders_desc)

if __name__ == '__main__':
   main_loop()




