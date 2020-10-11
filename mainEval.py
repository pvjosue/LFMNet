import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
from torch import optim
import torchvision.models as models
from torch.autograd import Variable
import torchvision as tv
import random
import math
import time
from datetime import datetime
import os
import argparse
import subprocess
from util.LFUtil import *
import numpy as np

from networks.LFMNet import LFMNet

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# Arguments
parser = argparse.ArgumentParser()
# Image indices to use for training and validation
parser.add_argument('--imagesToUse', nargs='+', type=int, default=list(range(0,15,1)))
# GPUs to use
parser.add_argument('--GPUs', nargs='+', type=int, default=[0])
# Path to dataset
parser.add_argument('--datasetPath', nargs='?', default="BrainLFMConfocalDataset/Brain_40x_64Depths_362imgs.h5")
# Path to directory where models and tensorboard logs are stored
parser.add_argument('--outputPath', nargs='?', default="eval/runsMouse/")
# Path to model to use for testing
parser.add_argument('--checkpointPath', nargs='?', default="runs/2020_10_11__14:23:21_TrueB_0.1bias_5I_128BS_FalseSk_9FOV_3nT_0.03ths_a8d9a2c_commit_")
# File to use
parser.add_argument('--checkpointFileName', nargs='?', default="model_130")
# Write volumes to H5 file
parser.add_argument('--writeVolsToH5', type=str2bool, default=False)
# Write output to tensorboard
parser.add_argument('--writeToTB', type=str2bool, default=True)

argsTest = parser.parse_args()
nImgs = len(argsTest.imagesToUse)

# Setup multithreading
num_workers = 0

if not torch.cuda.is_available():
        print("GPU initialization error")
        exit(-1)

# Select GPUs to use 
argsTest.GPUs = list(range(torch.cuda.device_count())) if argsTest.GPUs is None else argsTest.GPUs
print('Using GPUs: ' + str(argsTest.GPUs))

# Load checkpoint if provided
if argsTest.checkpointPath is not None:
    checkpointPath = argsTest.checkpointPath + "/" + argsTest.checkpointFileName
    checkpoint = torch.load(checkpointPath)
    # overwrite args
    argsModel = checkpoint['args']
    argsModel.checkpointPath = checkpointPath

# set Device to use
device = torch.device("cuda:"+str(argsTest.GPUs[0]) if torch.cuda.is_available() else "cpu")

# Create output folder
save_folder = argsTest.outputPath + argsTest.checkpointPath[:-1].split('/')[1] + "_eval_" + datetime.now().strftime('%Y_%m_%d__%H:%M:%S')
print(save_folder)

# Create summary writer to log stuff
if argsTest.writeToTB:
    writer = SummaryWriter(log_dir=save_folder)

# Load dataset
all_data = Dataset(argsTest.datasetPath, argsModel.randomSeed, \
    fov=argsModel.fovInput, neighShape=argsModel.neighShape, img_indices=argsTest.imagesToUse, get_full_imgs=True, center_region=None)
# Create data loader
test_dataset = data.DataLoader(all_data, batch_size=1,
                               shuffle=False, num_workers=num_workers, pin_memory=True)

# Get Dataset information
nDepths = all_data.get_n_depths()
volShape, LFshape = all_data.__shape__()
LFshape = LFshape[0:4]
lateralTile = int(math.sqrt(nDepths))
# Find normalization values
maxInputTrain, maxVolumeTrain = all_data.get_max()
maxInputTest, maxVolumeTest = all_data.get_max()

# Create network
net = LFMNet(nDepths, argsModel.useBias, argsModel.useSkipCon, LFshape, LFfov=argsModel.fovInput, use_small_unet=argsModel.useShallowUnet).to(device)
lossFunction = nn.L1Loss()
lossFunction.eval()
# Create SSIM criteria
ssim = SSIM()
ssim.eval()

# Start distributed data parallel, as it's faster than DataParallel
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1234'+str(argsTest.GPUs[0])
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1) #initialize torch.distributed 

# Move network to distributed data parallel
net = nn.parallel.DistributedDataParallel(net, device_ids=argsTest.GPUs, output_device=argsTest.GPUs[0]).to(device)
# Load network from checkpoint
net.load_state_dict(checkpoint['model_state_dict'])

# Move net to single GPU
net = net.module.to("cuda:1")
device = "cuda:1"
# timers
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

print('Testing')
net.eval()
avg_psnr = 0
avg_ssim = 0
avg_loss = 0
avg_time = 0

with torch.no_grad():
    # Evaluate images
    for nBatch,(inputs,labels) in enumerate(test_dataset):
        inputGPU = inputs.float().to(device) / maxInputTest
        outputsGT = labels.float().to(device) / maxVolumeTrain
        # Threshold GT to get rid of autofluorescence
        outputsGT = imadjust(outputsGT,argsModel.ths,outputsGT.max(), outputsGT.min(), outputsGT.max())


        start.record()
        outputsVol = net(inputGPU)
        end.record()
        torch.cuda.synchronize()
        curr_time = start.elapsed_time(end)

        curr_loss = lossFunction(outputsGT,outputsVol).item()
        avg_loss += curr_loss  / len(test_dataset)
        # Compute PSNR
        lossMSE = nn.functional.mse_loss(outputsVol.detach(), outputsGT.to(device).detach())
        curr_psnr = 10 * math.log10(1 / lossMSE.item())
        avg_psnr += curr_psnr / len(test_dataset)
        curr_ssim = ssim(outputsVol[:,0,:,:,:].permute(0,3,1,2).contiguous().detach(), outputsGT[:,0,:,:,:].permute(0,3,1,2).contiguous().to(device).detach()).sum().item()
        avg_ssim += curr_ssim / len(test_dataset)

        avg_time += curr_time / len(test_dataset)

        if argsTest.writeVolsToH5:
            h5file = h5py.File(save_folder+"/ReconVol_"+argsTest.checkpointFileName+'_'+str(nBatch+min(argsTest.imagesToUse))+".h5", 'w')
            h5file.create_dataset("LF4D", data=inputGPU.detach().cpu().squeeze().numpy())
            h5file.create_dataset("LFimg", data=LF2Spatial(inputGPU, inputGPU.shape[2:]).squeeze().cpu().detach().numpy())

            h5file.create_dataset("GT", data=outputsGT.detach().cpu().squeeze().numpy())
            h5file.create_dataset("reconFull", data=outputsVol.detach().cpu().squeeze().numpy())
            h5file.close()        



        if argsTest.writeToTB:
            curr_it = nBatch
            lastBatchSize = 1
            gridOut2 = torch.cat((outputsGT[0:lastBatchSize, :, :, :, :].sum(2).cpu().data.detach(), outputsVol[0:lastBatchSize, :, :, :, :].sum(2).cpu().data.detach()), dim=0)
            gridOut2 = tv.utils.make_grid(gridOut2, normalize=True, scale_each=False)
            LFImage = LF2Spatial(inputGPU, inputGPU.shape[2:])

            writer.add_image('images_val_YZ_projection', gridOut2, curr_it)
            z_proj = outputsGT[0,:,:,:,:].sum(3)
            writer.add_image('z_proj_GT',(z_proj/z_proj.max()).detach().cpu(),curr_it)
            z_proj = outputsVol[0,:,:,:,:].sum(3)
            writer.add_image('z_proj_prediction',(z_proj/z_proj.max()).detach().cpu(),curr_it)
            writer.add_image('LFImage_in', LFImage[0,:,:,:], curr_it)
            writer.add_scalar('Loss/test', curr_loss, curr_it)
            writer.add_scalar('Loss/psnr', curr_psnr, curr_it)
            writer.add_scalar('Loss/ssim', curr_ssim, curr_it)   
            writer.add_scalar('times/val', curr_time, curr_it)  
        
        print('Img: ' + str(nBatch) + '/' + str(len(test_dataset)) + " L1: " + str(curr_loss) + " psnr: " + str(curr_psnr) + " SSIM: " + str(curr_ssim) + " recon_time: " + str(curr_time))

print("avg_loss: " + str(avg_loss) + " avg_psnr: " + str(avg_psnr) + " avg_ssim: " + str(avg_ssim) + " avg_time: " + str(avg_time) + "ms")
writer.close()
    

