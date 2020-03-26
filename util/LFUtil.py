import tables
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
import torchvision as tv
import random
import h5py
from math import exp
import os
import sys
class Dataset(object):
    def __init__(self, fname, randomSeed=None, img_indices=None, fov=None, \
         neighShape=1, keep_imgs=False, random_imgs=False, center_region=None, get_full_imgs=False):
        self.fname = fname
        self.tables = tables.open_file(self.fname, "r")#, driver="H5FD_CORE")
        self.neighShape = neighShape
        self.LFShape = self.tables.root.LFData.shape
        self.volShape = self.tables.root.volData.shape
        self.tilesPerImage = self.LFShape[2:4]
        self.keep_imgs = keep_imgs
        self.nPatchesPerImg = self.tilesPerImage[0]*self.tilesPerImage[1]
        self.randomSeed = randomSeed
        self.nImagesInDB = self.LFShape[-1]
        self.getFullImgs = get_full_imgs
        self.fov = fov
        self.nDepths = self.volShape[2]
        if randomSeed is not None:
            torch.manual_seed(randomSeed)
        if fov is None:
            fov = 9
        self.centerRegion = center_region

        # Defines ranges to use accross each dimension
        LF_ix_y = list(range(0,self.LFShape[2]))
        LF_ix_x = list(range(0,self.LFShape[3]))
        vol_ix_y = list(range(0,self.volShape[0]))
        vol_ix_x = list(range(0,self.volShape[1]))

        # If a center region of the images is desired, crop
        if center_region is not None:
            LF_ix_y = list(range(self.LFShape[2]//2-center_region//2,self.LFShape[2]//2+center_region//2))
            LF_ix_x = list(range(self.LFShape[3]//2-center_region//2,self.LFShape[3]//2+center_region//2))
            vol_ix_y = list(range(self.volShape[0]//2-(center_region*self.LFShape[0])//2,self.volShape[0]//2+(center_region*self.LFShape[0])//2))
            vol_ix_x = list(range(self.volShape[1]//2-(center_region*self.LFShape[1])//2,self.volShape[1]//2+(center_region*self.LFShape[1])//2))
            self.tilesPerImage = [center_region, center_region]
            self.nPatchesPerImg = self.tilesPerImage[0]*self.tilesPerImage[1]
            self.LFShape = tuple([self.LFShape[0],self.LFShape[1],center_region, center_region,self.LFShape[-1]])
            self.volShape = tuple([self.LFShape[0]*center_region, self.LFShape[1]*center_region, self.volShape[-2], self.volShape[-1]])

        self.LFSideLenght = fov + neighShape - 1
        self.VolSideLength = self.neighShape * self.LFShape[0]

        # Use images either suggested by user or all images
        if img_indices is None:
            self.img_indices = range(0,self.nImagesInDB)
        else:
            self.img_indices = img_indices
        
        # Randomize images
        if random_imgs:
            self.img_indices = torch.randperm(int(self.nImagesInDB))

        self.nImagesToUse = len(self.img_indices)

        self.nPatches = self.nPatchesPerImg * (self.nImagesToUse)

        # Compute padding
        fov_half = self.fov//2
        neighShapeHalf = self.neighShape//2
        startOffset = fov_half + neighShapeHalf
        paddedLFSize = self.LFShape[:2] + tuple([self.LFShape[2]+2*startOffset,self.LFShape[3]+2*startOffset])
        paddedVolSize = tuple([self.volShape[0]+2*neighShapeHalf*self.LFShape[0],self.volShape[1]+2*neighShapeHalf*self.LFShape[1],self.volShape[2]])
        self.LFFull = torch.zeros(paddedLFSize+tuple([self.nImagesToUse]),dtype=torch.uint8)
        self.VolFull = torch.zeros(paddedVolSize+tuple([self.nImagesToUse]),dtype=torch.uint8)

        
        print("Loading img:  ",end=' ')
        for nImg,imgIx in enumerate(self.img_indices):
            print(str(imgIx),end=' ')
            # Load data from database
            currLF = torch.tensor(self.tables.root.LFData[:,:,:,:,imgIx], dtype=torch.uint8)
            currVol = torch.tensor(self.tables.root.volData[:,:,:,imgIx], dtype=torch.uint8)
            currLF = currLF[:,:,LF_ix_y,:]
            currLF = currLF[:,:,:,LF_ix_x]
            currVol = currVol[vol_ix_y,:,:]
            currVol = currVol[:,vol_ix_x,:]
            # Pad with zeros borders
            currLF = F.pad(currLF, (startOffset, startOffset, startOffset, startOffset, 0, 0, 0, 0))
            currVol = F.pad(currVol, (0,0,neighShapeHalf*self.LFShape[1],neighShapeHalf*self.LFShape[1],\
                neighShapeHalf*self.LFShape[0],neighShapeHalf*self.LFShape[0]))
            self.LFFull[:,:,:,:,nImg] = currLF
            self.VolFull[:,:,:,nImg] = currVol
                
        self.volMax = self.VolFull.max()
        self.LFMax = self.LFFull.max()
        self.VolDims = [neighShape*self.LFShape[0],neighShape*self.LFShape[1],self.volShape[2]]
        self.LFDims = [self.LFShape[0],self.LFShape[1],self.LFSideLenght,self.LFSideLenght]

        if self.getFullImgs:
            self.VolDims = self.volShape[0:3]#+tuple([len(self.img_indices)])
            self.LFDims = self.LFShape[0:4]#+tuple([len(self.img_indices)])
            self.nPatches = len(self.img_indices)
        self.tables.close()

    def __getitem__(self, index):
        # Fetch full image or patches
        if self.getFullImgs:
            currLFPatch = self.LFFull[:,:,:,:, index].unsqueeze(0)
            currVolPatch = self.VolFull[:,:,:, index].unsqueeze(0)
        else:
            nImg = index//self.nPatchesPerImg
            nPatch = index - nImg*self.nPatchesPerImg
            yLF = nPatch//self.LFShape[3]
            xLF = nPatch%self.LFShape[3]
            yVol = yLF*self.LFShape[0]
            xVol = xLF*self.LFShape[1]

            # Crop current patch
            currLFPatch = self.LFFull[:,:,yLF:yLF+self.LFSideLenght, xLF:xLF+self.LFSideLenght, nImg].unsqueeze(0)
            currVolPatch = self.VolFull[yVol:yVol+self.VolSideLength, xVol:xVol+self.VolSideLength,:, nImg].unsqueeze(0)
        
        return currLFPatch, currVolPatch

    def __len__(self):
        return self.nPatches

    def get_n_depths(self):
        return self.VolDims[-1]
    def get_max(self):
        return self.LFMax, self.volMax
    def __shape__(self):
        return self.VolDims, self.LFDims
     
def convert3Dto2DTiles(x, lateralTile):
    nDepths = x.shape[-1]
    volSides = x.shape[-3:-1]
    nChans = x.shape[1]
    verticalTile = x.permute(0, 1, 4, 2, 3).contiguous().view(-1, nChans, volSides[0]*nDepths, volSides[1])
    currPred = verticalTile[:,:,0:volSides[0]*lateralTile[0],:]
    for k in range(1,lateralTile[1]):
        currPred = torch.cat((currPred, verticalTile[:,:,(lateralTile[0]*volSides[0]*k):(lateralTile[0]*volSides[0]*(k+1)),:]), dim=3)
    return currPred

def convert4Dto3DTiles(x, lateralTile):
    nDepths = x.shape[-1]
    volSide = x.shape[-2]
    nSamples = x.shape[0]
    verticalTile = x.permute(1,0,2,3).contiguous().view(volSide,volSide*nSamples,nDepths)
    currPred = verticalTile[:,0:volSide*lateralTile[0],:]
    for k in range(1,lateralTile[1]):
        currPred = torch.cat((currPred, verticalTile[:,(lateralTile[0]*volSide*k):(lateralTile[0]*volSide*(k+1)),:]), dim=0)
    return currPred

def LF2Spatial(xIn, LFSize):
    xShape = xIn.shape
    x = xIn
    if xIn.ndimension() == 6:
        x = xIn.permute((0,1,4,2,5,3)).contiguous().view(xShape[0], xShape[1], LFSize[0] * LFSize[2], LFSize[1] * LFSize[3])
    if xIn.ndimension() == 4:
        x = xIn.view(xShape[0],xShape[1],LFSize[2],LFSize[0],LFSize[3],LFSize[1]).permute((0,1,3,5,2,4)).contiguous()
    return x

def LF2Angular(xIn, LFSize):
    xShape = xIn.shape
    x = xIn
    if xIn.ndimension() == 6:
        x = xIn.permute((0,1,2,4,3,5)).contiguous().view(xShape[0], xShape[1], LFSize[0] * LFSize[2], LFSize[1] * LFSize[3])
    if xIn.ndimension() == 4:
        x = xIn.view(xShape[0],xShape[1],LFSize[0],LFSize[2],LFSize[1],LFSize[3]).permute((0,1,2,4,3,5)).contiguous()
    return x

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight.data, 1/len(m.weight.data))
    if isinstance(m, nn.ConvTranspose2d):
        nn.init.constant_(m.weight.data, 1/len(m.weight.data))

def _getThreads():
    """ Returns the number of available threads on a posix/win based system """
    if sys.platform == 'win32':
        return (int)(os.environ['NUMBER_OF_PROCESSORS'])
    else:
        return (int)(os.popen('grep -c cores /proc/cpuinfo').read())

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    mask = (y>0).float()
    y = torch.mul(y,mask)
    return y




######## SSIM

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)