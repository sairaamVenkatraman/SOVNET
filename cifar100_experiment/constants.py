import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
STRINGDEVICE = "cuda:1"
LR = 0.001
