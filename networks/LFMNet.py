import torch
import torch.nn as nn
from util.LFClases import *
from util.TorchNDFunctions.FunctionsNd import ConvNd

class LFMNet(nn.Module):
    def __init__(self, nDepths, useBias, useSkipCon, LFshape, LFfov=9, use_small_unet=False):
        super(LFMNet, self).__init__()
        self.nDepths = nDepths
        self.LFshape = LFshape
        
        if use_small_unet:
            from networks.UnetShallow import UNetLF
        else:
            from networks.UnetFull import UNetLF

        self.lensletConvolution = nn.Sequential(
            ConvNd(1,self.nDepths, num_dims=4, kernel_size=(3,3,LFfov,LFfov),stride=1, padding=(1,1,0,0), use_bias=useBias),
            nn.LeakyReLU())
        
        self.Unet = UNetLF(self.nDepths, self.nDepths, use_skip=useSkipCon)   

    def forward(self, input):
        # 4D convolution
        inputAfter4DConv = self.lensletConvolution(input)
        # 4D to 2D image
        newLFSize = inputAfter4DConv.shape[2:]
        newLensletImage = LF2Spatial(inputAfter4DConv, newLFSize)
        # U-net prediction
        x = self.Unet(newLensletImage)
        # Channels to 3D dimension 
        x3D = x.permute((0,2,3,1)).unsqueeze(1)
        
        return x3D