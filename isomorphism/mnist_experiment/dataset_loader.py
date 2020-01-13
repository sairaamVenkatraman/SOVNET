from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATAPATH = "../../data/MNIST/"
BATCHSIZE = 64

def get_loaders_mnist(train_translation_rotation_list,test_translation_rotation_list,batch_size):
    """Load mnist dataset. The data is divided by 255 and subracted by mean and divided by standard deviation.
    """
    train_loaders_desc = []
    test_loaders_desc = []
    for (translation,rotation) in train_translation_rotation_list:
        train_desc = 'train_'+str(translation[0])+'_'+str(translation[1])+'_'+str(rotation)+'_'+'MNIST'#for identifying in logs 
        train_transform = transforms.Compose([
                                        transforms.Resize(29),
                                        #transforms.RandomAffine(rotation,translation),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        ])
        training_set = MNIST(DATAPATH, train=True, download=True, transform=train_transform)
        training_data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        train_loaders_desc.append((training_data_loader,train_desc))
    for (translation,rotation) in test_translation_rotation_list:
        test_desc = 'test_'+str(translation[0])+'_'+str(translation[1])+'_'+str(rotation)+'_'+'MNIST'#for identifying in logs  
        test_transform = transforms.Compose([
                                        transforms.Resize(29),
                                        transforms.RandomRotation(degrees=(rotation,rotation)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        ])
        testing_set = MNIST(DATAPATH, train=False, download=True, transform=test_transform)  
        testing_data_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True)
        test_loaders_desc.append((testing_data_loader,test_desc)) 
    return train_loaders_desc, test_loaders_desc
