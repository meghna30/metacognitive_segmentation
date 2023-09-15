import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class EncBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.pool(x_conv)

        return x_conv, x_pool



class DecBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecBlock, self).__init__()
        self.up = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        # nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0),
        # nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.conv = ConvBlock(out_channels*2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis = 1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, input_shape):
        super(UNet, self).__init__()

        "Encoder"

        self.enc1 = EncBlock(in_channels, 128)
        self.enc2 = EncBlock(128,256)
        self.enc3 = EncBlock(256,512)
        #self.enc4 = EncBlock(256, 512)
        #self.enc5 = EncBlock(512,1024)
        "Bottleneck"
        self.bottleneck = ConvBlock(512,1024)

        "Decoder"
        self.dec1 = DecBlock(1024, 512)
        self.dec2 = DecBlock(512, 256)
        self.dec3 = DecBlock(256, 128)
        #self.dec4 = DecBlock(256, 128)
        #self.dec5 = DecBlock(128, 64)

        self.outputs = nn.Sequential(
                        nn.Conv2d(128,1,kernel_size = 1, padding = 0),
                        )
        self.input_shape = input_shape

    def forward(self, x):
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        #skip4, x = self.enc4(x)
        #skip5, x = self.enc5(x)
        x = self.bottleneck(x)

        #x = self.dec1(x, skip5)
        #x = self.dec2(x, skip4)
        x = self.dec1(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec3(x, skip1)
        out = self.outputs(x)
        # out = F.interpolate(x, size=self.input_shape, mode='bilinear', align_corners=False)

        return out
