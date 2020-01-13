# SOVNET
This repository contains code for the ICLR submission "Building Deep Equivariant Capsule Networks"
Each folder contains pytorch code for the model, as well as the training details for experiments on various datasets. Thus, the folder 
"svhn_experiment" contains code for the experiments done on SVHN for SOVNET.

We have performed transformation-robustness experiments on MNIST, FashionMNIST, and CIFAR-10. We also tested the performance of SOVNET
on augmented versions of SVHN and KMNIST.

Note that the code has been written for use on multi-gpu - this can be changed by commenting out the nn.DataParallel() calls. Also,
please install the package for group-equivariant convolutions. This can be obtained from https://github.com/adambielski/GrouPy
