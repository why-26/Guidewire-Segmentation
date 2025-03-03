import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from torchvision.ops import DeformConv2d
from timm.models.layers import trunc_normal_
from .attention.agent_swin import EfficientMixAttnTransformerBlock,PatchExpand
import torch.nn.functional as F


class DepthWiseConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,padding=1,stride=1,dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,padding=padding,
                               stride=stride,dilation=dilation,groups=in_channels)
        self.norm_layer = nn.BatchNorm2d(in_channels) 
        self.conv2 = nn.Conv2d(in_channels,out_channels,kernel_size=1)
    
    def forward(self,x):
        return self.conv2(self.norm_layer(self.conv1(x)))
 

class Shift8(nn.Module):
    def __init__(self, groups=4, stride=1, mode='constant') -> None:
        super().__init__()
        self.g = groups
        self.mode = mode
        self.stride = stride

    def forward(self, x):
        b, c, h, w = x.shape
        out = torch.zeros_like(x)

        pad_x = F.pad(x, pad=[self.stride for _ in range(4)], mode=self.mode)
        assert c == self.g * 8

        cx, cy = self.stride, self.stride
        stride = self.stride
        out[:,0*self.g:1*self.g, :, :] = pad_x[:, 0*self.g:1*self.g, cx-stride:cx-stride+h, cy:cy+w]
        out[:,1*self.g:2*self.g, :, :] = pad_x[:, 1*self.g:2*self.g, cx+stride:cx+stride+h, cy:cy+w]
        out[:,2*self.g:3*self.g, :, :] = pad_x[:, 2*self.g:3*self.g, cx:cx+h, cy-stride:cy-stride+w]
        out[:,3*self.g:4*self.g, :, :] = pad_x[:, 3*self.g:4*self.g, cx:cx+h, cy+stride:cy+stride+w]

        out[:,4*self.g:5*self.g, :, :] = pad_x[:, 4*self.g:5*self.g, cx+stride:cx+stride+h, cy+stride:cy+stride+w]
        out[:,5*self.g:6*self.g, :, :] = pad_x[:, 5*self.g:6*self.g, cx+stride:cx+stride+h, cy-stride:cy-stride+w]
        out[:,6*self.g:7*self.g, :, :] = pad_x[:, 6*self.g:7*self.g, cx-stride:cx-stride+h, cy+stride:cy+stride+w]
        out[:,7*self.g:8*self.g, :, :] = pad_x[:, 7*self.g:8*self.g, cx-stride:cx-stride+h, cy-stride:cy-stride+w]

        return out


class Shift_Deformable_Attention(nn.Module):
    def __init__(self, channels):
        super(Shift_Deformable_Attention, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),  
            Shift8(groups=channels//8, stride=1),  
            nn.InstanceNorm2d(channels),
            nn.GELU()
        )

        self.offset_channels = 2 * 3 * 3  
        self.offset_conv = nn.Conv2d(channels, self.offset_channels, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(channels, channels, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_reduced = self.bottleneck(x)
        offset = self.offset_conv(x_reduced)
        x_deform = self.deform_conv(x_reduced, offset)
        x_norm_shape = x_deform.size()[1:]  
        x_norm = F.layer_norm(x_deform, x_norm_shape)  
        attention = self.sigmoid(x_norm)
        x_out = x * attention 

        return x_out
        
class IC(nn.Module):
    def __init__(self, num_channels, dropout_prob=0.2):
        super(IC, self).__init__()
        self.bn = nn.BatchNorm2d(num_channels)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.bn(x)
        x = self.dropout(x)
        return x 
        
                    
# 普通卷积
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), # 设置padding = 1，特征图不会在进行卷积时改变
            IC(mid_channels),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            IC(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Double_GCT_Conv(nn.Module):
    def __init__(self, in_channels,out_channels,mid_channels=None,dirate=1):
        super(Double_GCT_Conv,self).__init__()
        self.gate = Shift_Deformable_Attention(in_channels)
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), # 设置padding = 1，特征图不会在进行卷积时改变
            IC(mid_channels),
            nn.LeakyReLU(inplace=True), 
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            IC(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    
    def forward(self,x):
        x_pro = self.gate(x)
        return self.double_conv(x_pro)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,attn=False):
        super().__init__()
        if attn:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                Double_GCT_Conv(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )
    def forward(self, x):
        return self.maxpool_conv(x)

# Bilateral Guided Aggregation 
class BGALayer(nn.Module):
    """" 
        5skip connnection
    """    

    def __init__(self,in_channels,out_channels):
        super(BGALayer,self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1,groups=in_channels,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,padding=0,bias=False)
        )
        # 3*3 Conv 
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=2,padding=1,bias=False), # (H/2,W/2,C)
            nn.BatchNorm2d(in_channels)
        )
        # 3*3 Conv 
        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1,bias=False), # (H/2,W/2,C)
            nn.BatchNorm2d(in_channels),
        )
        # 3*3 DWConv + 1*1 
        self.decoder2 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1,groups=in_channels,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,padding=0,bias=False)
        )
        # sum--->conv
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),    
        )
        
    def forward(self,x_low,x_high):
        low_size = x_low.size()[2:] 
        high_size = x_high.size()[2:]

        encoder_low1 = self.encoder1(x_low) 
        encoder_low2 = self.encoder2(x_low) 
        decoder_high1 = self.decoder1(x_high) 
        decoder_high2 = self.decoder2(x_high) 
            
        decoder_high1 = F.interpolate(decoder_high1,size=low_size,mode='bilinear',align_corners=True)
        low = encoder_low1 * torch.sigmoid(decoder_high1)
        high = encoder_low2 * torch.sigmoid(decoder_high2)
        high = F.interpolate(high,size=low_size,mode='bilinear',align_corners=True)
        out = self.out(low + high)
        return out
    

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True,if_first=False,attn=False):
        super().__init__()

        self.if_first = if_first
        self.pointwise = nn.Conv2d(in_channels*2,in_channels,kernel_size=1,stride=1,padding=0)
        self.conv1 = Double_GCT_Conv(in_channels*2,in_channels)
        if  bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if attn:
                self.conv = Double_GCT_Conv(in_channels*2, out_channels, in_channels) # 1024,512,256
            else:
                self.conv = DoubleConv(in_channels*2, out_channels, in_channels) # 因为是拼接的通道，第二次变为输入的一半 64 32 32
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            if attn:
                self.conv = Double_GCT_Conv(in_channels*2, out_channels, in_channels) # 1024,512,256
            else:
                self.conv = DoubleConv(in_channels*2, out_channels, in_channels) # 因为是拼接的通道，第二次变为输入的一半 64 32 32

        self.bga = BGALayer(in_channels,in_channels) 
    def forward(self, x1, x2):
        x3 = self.up(x1) 
        bga = self.bga(x2,x1)
        x = torch.cat([bga, x3], dim=1) 
        out = self.conv(x) 
        return out


# 最后输出
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
