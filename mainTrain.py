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

# Arguments
parser = argparse.ArgumentParser()
# Number of epochs
parser.add_argument('--epochs', type=int, default=1000)
# Validate every n percentage of the data
parser.add_argument('--valEvery', type=float, default=0.25)
# Image indices to use for training and validation
parser.add_argument('--imagesToUse', nargs='+', type=int, default=list(range(0,5,1)))
# List of GPUs to use: 0 1 2 for example
parser.add_argument('--GPUs', nargs='+', type=int, default=None)
# Batch size
parser.add_argument('--batchSize', type=int, default=128)
# Perentage of the data to use for validation, from 0 to 1
parser.add_argument('--validationSplit', type=float, default=0.1)
# Bias initialization value
parser.add_argument('--biasVal', type=float, default=0.1)
# Learning rate
parser.add_argument('--learningRate', type=float, default=0.005)
# Use bias flag
parser.add_argument('--useBias', type=str2bool, default=True)
# Use skip connections flag
parser.add_argument('--useSkipCon', type=str2bool, default=False)
# User selected random seed
parser.add_argument('--randomSeed', type=int, default=None) 
# fov of input or neighboarhood around lenslet to reconstruct
parser.add_argument('--fovInput', type=int, default=9)
# nT number of lenslets to reconstruct simultaneously use at training time
parser.add_argument('--neighShape', type=int, default=3)
# Flag to use shallow or large U-net
parser.add_argument('--useShallowUnet', type=str2bool, default=True)
# Lower threshold of GT stacks, to get rid of autofluorescence
parser.add_argument('--ths', type=float, default=0.03)
# Path to dataset
parser.add_argument('--datasetPath', nargs='?', default="BrainLFMConfocalDataset/Brain_40x_64Depths_362imgs.h5")
# Path to directory where models and tensorboard logs are stored
parser.add_argument('--outputPath', nargs='?', default="runs/")
# Prefix for current output folder
parser.add_argument('--outputPrefix', nargs='?', default="")
# Path to model in case of continuing a training
parser.add_argument('--checkpointPath', nargs='?', default=None)

args = parser.parse_args()
nImgs = len(args.imagesToUse)

# Setup multithreading
num_workers = getThreads()
if num_workers!=0:
    torch.set_num_threads(num_workers)

if not torch.cuda.is_available():
        print("GPU initialization error")
        exit(-1)

# Select GPUs to use 
args.GPUs = list(range(torch.cuda.device_count())) if args.GPUs is None else args.GPUs
print('Using GPUs: ' + str(args.GPUs))
device_ids = args.GPUs

# Set common random seed
if args.randomSeed is not None:
    np.random.seed(args.randomSeed)
    torch.manual_seed(args.randomSeed)

# Load checkpoint if provided
if args.checkpointPath is not None:
    checkpointPath = args.checkpointPath
    checkpoint = torch.load(checkpointPath)
    # overwrite args
    args = checkpoint['args']
    args.checkpointPath = checkpointPath

# set Device to use
device = torch.device("cuda:"+str(device_ids[0]) if torch.cuda.is_available() else "cpu")

# Create unique label
today = datetime.now()
# Get commit number 
label = subprocess.check_output(["git", "describe", "--always"]).strip()
comment = today.strftime('%Y_%m_%d__%H:%M:%S') + "_"+ str(args.useBias) +"B_"+str(args.biasVal)+"bias_" + str(nImgs) + \
     "I_"+ str(args.batchSize)+"BS_"+str(args.useSkipCon)+"Sk_" +  str(args.fovInput) + "FOV_" + str(args.neighShape) + "nT_" \
            + str(args.ths) + "ths_" + str(label.decode("utf-8") ) + "_commit__" + args.outputPrefix

# Create output folder
save_folder = args.outputPath + "/" + comment
# If asked to continue a training, save in the same folder
if args.checkpointPath is not None:
    save_folder = os.path.split(args.checkpointPath)[0]
print(save_folder)

# Create summary writer to log stuff
writer = SummaryWriter(log_dir=save_folder)
writer.add_text('Description',comment,0)
writer.flush()




# Load dataset
all_data = Dataset(args.datasetPath, args.randomSeed, \
    fov=args.fovInput, neighShape=args.neighShape, img_indices=args.imagesToUse, get_full_imgs=False, center_region=None)
# Split validation and testing
train_size = int((1 - args.validationSplit) * len(all_data))
test_size = len(all_data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(all_data, [train_size, test_size])
# Create data loaders
train_dataset = data.DataLoader(train_dataset, batch_size=args.batchSize,
                                shuffle=True, num_workers=num_workers, pin_memory=True)
test_dataset = data.DataLoader(test_dataset, batch_size=args.batchSize,
                               shuffle=True, num_workers=num_workers, pin_memory=True)

validate_every = np.round(len(train_dataset)*args.valEvery)

# Get Dataset information
nDepths = all_data.get_n_depths()
volShape, LFshape = all_data.__shape__()
LFshape = LFshape[0:4]
lateralTile = int(math.sqrt(nDepths))
# Find normalization values
maxInputTrain, maxVolumeTrain = all_data.get_max()
maxInputTest, maxVolumeTest = all_data.get_max()

# Create network
net = LFMNet(nDepths, args.useBias, args.useSkipCon, LFshape, LFfov=args.fovInput, use_small_unet=args.useShallowUnet).to(device)
optimizer = optim.Adam(net.parameters(), lr=args.learningRate)
lossFunction = nn.L1Loss()
# Create SSIM criteria
ssim = SSIM()
ssim.eval()

# Init bias and weights if needed
if args.useBias:
    def bias_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            if m.bias is not None:
                nn.init.constant_(m.bias.data, args.biasVal)
            nn.init.kaiming_normal_(m.weight)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.constant_(m.bias.data, args.biasVal)
            nn.init.kaiming_normal_(m.weight)
    net.apply(bias_init)

# Load network from checkpoint
if args.checkpointPath is not None:
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochStart = checkpoint['epoch']
    epochs = args.epochs + epochStart
    train_loss = checkpoint['loss']


# Start distributed data parallel, as it's faster than DataParallel
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1234'+str(device_ids[0])
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

    # Move network to distributed data parallel
    net = nn.parallel.DistributedDataParallel(net, device_ids=args.GPUs, output_device=args.GPUs[0]).to(device)


# timers
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
global_it_counter = 0
# define indices to grab for tensorboard visualization
indices_to_show = torch.randperm(test_size)[0:8]
# Init arrays to store losses
train_losses, test_losses = [], []
test_loss = 0
epochStart = 0

# Start training
for epoch in range(epochStart, args.epochs):
    net.train()
    torch.set_grad_enabled(True)
    train_loss = 0
    print('Training')
    global_it_counter = 0

    for nBatch,(inputs,labels) in enumerate(train_dataset):
        # compute current iteration
        curr_it = epoch*len(train_dataset) + nBatch
        # start timer
        start.record()
        print('ep: ' + str(epoch) + '  ' + str(nBatch+1) + '/' + str(len(train_dataset)) + ' currIt: ' + str(curr_it))

        optimizer.zero_grad() 
        # load data to gpu and normalize from 0 to 1
        inputGPU = inputs.float().to(device) / maxInputTest
        outputsGT = labels.float().to(device) / maxVolumeTrain
        # Threshold GT to get rid of autofluorescence
        if args.ths!=0:
            outputsGT = imadjust(outputsGT, args.ths,outputsGT.max(), outputsGT.min(), outputsGT.max())
        # Predict
        outputsVol = net(inputGPU)
        loss = lossFunction(outputsGT,outputsVol)
        loss.backward()
        train_loss += loss.item() / nDepths
        optimizer.step()

        global_it_counter += inputs.shape[0]
        # Record training time
        end.record()
        torch.cuda.synchronize()
        end_time = start.elapsed_time(end)
        # Compute time per sample
        elapsed_time = end_time/inputs.shape[0]

        # Check if validation is required
        if nBatch%validate_every==0:
            print(comment)
            # Write training images to tensorboard
            lastBatchSize = min(outputsGT.shape[0],4)
            gridOut2 = torch.cat((outputsGT[0:lastBatchSize, :, :, :, :].sum(2).cpu().data.detach(), outputsVol[0:lastBatchSize, :, :, :, :].sum(2).cpu().data.detach()), dim=0)
            gridOut2 = tv.utils.make_grid(gridOut2, normalize=True, scale_each=False)
            # Select some images in the batch for showing
            indices_to_display = torch.randperm(inputGPU.shape[0])[0:4]
            outputsGT = F.interpolate(outputsGT[indices_to_display, :, :, :, :],[LFshape[0]*2,LFshape[1]*2,volShape[2]])
            outputsVol = F.interpolate(outputsVol[indices_to_display, :, :, :, :],[LFshape[0]*2,LFshape[1]*2,volShape[2]])
            inputGPU = inputGPU[indices_to_display,:,:,:,:,:]
            currPred = convert3Dto2DTiles(outputsVol, [lateralTile, lateralTile])
            currGT = convert3Dto2DTiles(outputsGT, [lateralTile, lateralTile])
            inputGrid = LF2Spatial(inputGPU, inputGPU.shape[2:])
            gridPred = tv.utils.make_grid(currPred,normalize=True, scale_each=False)
            gridGT = tv.utils.make_grid(currGT,normalize=True, scale_each=False)
            gridInput = tv.utils.make_grid(inputGrid,normalize=True, scale_each=False)
            gt = outputsGT[0,:,:,:,:].sum(3).repeat(3,1,1)
            gt /= gt.max()
            # Write to tensorboard
            writer.add_image('z_proj_train',gt,curr_it)
            writer.add_image('images_train_YZ_projection', gridOut2, curr_it)        
            writer.add_image('outputRGB_train', gridPred, curr_it)
            writer.add_image('outputRGB_train_GT', gridGT, curr_it)
            writer.add_image('input_train', gridInput, curr_it)
            writer.add_scalar('Loss/train', train_loss/global_it_counter, curr_it)
            writer.add_scalar('times/train', elapsed_time, curr_it)
            
            # Restart
            train_loss = 0.0
            global_it_counter = 0


            print('Validating')
            net.eval()
            with torch.no_grad(): 
                avg_psnr = 0
                avg_ssim = 0
                test_loss = 0
                start.record()
                for nBatch,(inputs,labels) in enumerate(test_dataset):
                    inputGPU = inputs.float().to(device) / maxInputTest
                    outputsGT = labels.float().to(device) / maxVolumeTrain
                    # Threshold GT to get rid of autofluorescence
                    outputsGT = imadjust(outputsGT,args.ths,outputsGT.max(), outputsGT.min(), outputsGT.max())
                    outputsVol = net(inputGPU)
                    loss = lossFunction(outputsGT,outputsVol)
                    test_loss += loss.item() / nDepths
                    # Compute PSNR
                    lossMSE = nn.functional.mse_loss(outputsVol.to(device).detach(), outputsGT.to(device).detach())
                    avg_psnr += 10 * math.log10(1 / lossMSE.item())
                    # Compute ssim
                    avg_ssim += ssim(outputsVol[:,0,:,:,:].permute(0,3,1,2).contiguous().detach().to(device), outputsGT[:,0,:,:,:].permute(0,3,1,2).contiguous().detach().to(device)).sum()
                end.record()
                torch.cuda.synchronize()

            
                lastBatchSize = min(outputsGT.shape[0],4)
                gridOut2 = torch.cat((outputsGT[0:lastBatchSize, :, :, :, :].sum(2).cpu().data.detach(), outputsVol[0:lastBatchSize, :, :, :, :].sum(2).cpu().data.detach()), dim=0)
                gridOut2 = tv.utils.make_grid(gridOut2, normalize=True, scale_each=False)
                # process some for showing
                indices_to_display = torch.randperm(inputGPU.shape[0])[0:lastBatchSize]
                outputsGT = F.interpolate(outputsGT[indices_to_display, :, :, :, :],[LFshape[0]*2,LFshape[1]*2,volShape[2]])
                outputsVol = F.interpolate(outputsVol[indices_to_display, :, :, :, :],[LFshape[0]*2,LFshape[1]*2,volShape[2]])
                inputGPU = inputGPU[indices_to_display,:,:,:,:,:]

                currPred = convert3Dto2DTiles(outputsVol, [lateralTile, lateralTile])
                currGT = convert3Dto2DTiles(outputsGT, [lateralTile, lateralTile])
                inputGrid = LF2Spatial(inputGPU, inputGPU.shape[2:])
                gridPred = tv.utils.make_grid(currPred,normalize=True, scale_each=False)
                gridGT = tv.utils.make_grid(currGT,normalize=True, scale_each=False)
                gridInput = tv.utils.make_grid(inputGrid,normalize=True, scale_each=False)
                # Write to tensorboard
                writer.add_image('images_val_YZ_projection', gridOut2, curr_it)
                writer.add_image('outputRGB_test', gridPred, curr_it)
                writer.add_image('outputRGB_test_GT', gridGT, curr_it)
                writer.add_image('input_test', gridInput, curr_it)
                writer.add_scalar('Loss/test', test_loss/len(test_dataset), curr_it)
                writer.add_scalar('Loss/psnr_val', avg_psnr/len(test_dataset), curr_it)
                writer.add_scalar('Loss/ssim_val', avg_ssim/len(test_dataset), curr_it)    
                writer.add_scalar('LearningRate', args.learningRate, curr_it)
                writer.add_scalar('times/val', start.elapsed_time(end)/test_size, curr_it)  
            net.train()
    
    if epoch%2==0:
        torch.save({
        'epoch': epoch,
        'args' : args,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss, 
        'dataset_path': args.datasetPath},
        save_folder + '/model_'+str(epoch))

    print(f"Epoch {epoch + 1}/{args.epochs}.. "
            f"Train loss: {train_loss / len(train_dataset):.7f}.. "
            f"Test loss: {test_loss / len(test_dataset):.7f}.. ")

