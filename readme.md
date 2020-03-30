# LFMNet: A fully convolutional network for Light-field microscope (LFM) 3D reconstruction.

This repository contains the code from our LFMNet. A neural network that reconstructs a 3D confocal volume given a 4D LF image, it has been tested with the Mice Brain LFM-confocal [dataset](http://cvg.unibe.ch/media/project/page/LFMNet/index.html "LFMNet CVG project").
LFMNet is fully convolutional, it can be trained with LFs of any size (for example patches) and then tested on other sizes.
In our case it takes 20ms to reconstruct a volume with 1287x1287x64 voxels.

## Requirements
The repo is based on Python 3.7.4 and Pytorch 1.4, see requirements.txt for more details.
The dataset used for this network can be found [here](http://cvg.unibe.ch/media/project/page/LFMNet/index.html "LFMNet CVG project"), but it works with any LF image that has a corresponding 3D volume.

## Usage
  ### Input
  A tensor with shape **1,Ax,Ay,Sx,Sy**, where A are the angular dimensions and S the spatial dimensions. In our case the input tensor is **1,33,33,39,39**.
  ### Output
  A tensor with shape **nD,Ax*Sx,Ay*Sy**, where nD are the number of depths to reconstruct. In our case the output tensor is **64,1287,1287**.
    
  ### Train and test
  The training main file is mainTrain.py and mainEval.py the testing file.
  
## Network structure
The paradigm behind this network is that the input contains a group of microlenses and a neighborhood around them, and reconstructs the 3D volume behind the central microlenses.
  LFMNet has as an initial layer a [conv4d](https://github.com/pvjosue/TorchNDFunctions "4D convolution"), that ensures a fully convolutional network, this first layers traverses every lenslet, and grabs a neighborhood (9 lenses in our case) around. Then the output is converted to a 2D image with the number of channels equal to the number of depths to reconstruct. Lastly, this tensor goes into a U-net, which finishes up the feature extraction and 3D reconstution.
