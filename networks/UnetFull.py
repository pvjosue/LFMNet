# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode = 'reflect'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 2, padding=1, padding_mode = 'reflect'),
            double_conv(out_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False, use_skip=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            if use_skip:
                self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 3, stride=2, padding=1)
            else:
                self.up = nn.ConvTranspose2d(in_ch // 2, in_ch, 3, stride=2, padding=1)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2, use_skip=True):
        x = self.up(x1)
        if use_skip:
            diffY = x.size()[2] - x2.size()[2]
            diffX = x.size()[3] - x2.size()[3]

            x2 = F.pad(x2, (diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2))
            x = torch.cat((x,x2),1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, padding_mode = 'reflect')

    def forward(self, x):
        x = self.conv(x)
        return x



class UNetLF(nn.Module):
    def __init__(self, n_channels, n_classes, use_skip=True):
        super(UNetLF, self).__init__()
        self.use_skip = use_skip
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256, use_skip=use_skip)
        self.up2 = up(512, 128, use_skip=use_skip)
        self.up3 = up(256, 64, use_skip=use_skip)
        self.up4 = up(128, 64, use_skip=use_skip)
        self.outc = nn.Sequential(
            outconv(64, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        inputShape = x.shape
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4, use_skip=self.use_skip)
        x = self.up2(x, x3, use_skip=self.use_skip)
        x = self.up3(x, x2, use_skip=self.use_skip)
        x = self.up4(x, x1, use_skip=self.use_skip)
        x = F.interpolate(x, size=inputShape[-2:],mode='bilinear',align_corners=False)
        x = self.outc(x)
        return x