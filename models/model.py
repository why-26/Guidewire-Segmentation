from .model_parts import *

def initialize_weights(c):
    if isinstance(c,nn.Conv2d):
        init.normal_(c.weight.data,mean=0.0,std=0.01)
        if c.bias is not None:
            init.constant_(c.bias.data,0.0)
            
class FGANet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False,gt_ds=True):
        super(FGANet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.gt_ds = gt_ds
        self.inc = (DoubleConv(n_channels, 32)) 
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128,attn=True)) 
        self.down3 = (Down(128, 256,attn=True)) 
        self.down4 = (Down(256, 512,attn=True))
        self.down5 = (Down(512, 512,attn=True))

        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(512,1,1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(256,1,1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(128,1,1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(64,1,1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(32,1,1))
        
        self.up1 = (Up(512, 256, bilinear=True,attn=True)) 
        self.up2 = (Up(256, 128, bilinear=True,attn=True)) 
        self.up3 = (Up(128, 64, bilinear=True,attn=True))
        self.up4 = (Up(64, 32, bilinear=True)) 
        self.up5 = (Up(32, 32, bilinear=True)) 
        self.outc = (OutConv(32, n_classes)) 
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        if self.gt_ds:
            gt_pre5 = self.gt_conv1(x6)
            gt_pre5 = F.interpolate(gt_pre5,scale_factor=32,mode='bilinear',align_corners=True)
        x = self.up1(x6, x5)
        if self.gt_ds:
            gt_pre4 = self.gt_conv2(x)
            gt_pre4 = F.interpolate(gt_pre4,scale_factor=16,mode='bilinear',align_corners=True)
        x = self.up2(x, x4)
        if self.gt_ds:
            gt_pre3 = self.gt_conv3(x)
            gt_pre3 = F.interpolate(gt_pre3,scale_factor=8,mode='bilinear',align_corners=True)
        x = self.up3(x, x3)
        if self.gt_ds:
            gt_pre2 = self.gt_conv4(x)
            gt_pre2 = F.interpolate(gt_pre2,scale_factor=4,mode='bilinear',align_corners=True)
        x = self.up4(x, x2)
        if self.gt_ds:
            gt_pre1 = self.gt_conv5(x)
            gt_pre1 = F.interpolate(gt_pre1,scale_factor=2,mode='bilinear',align_corners=True)
        x = self.up5(x, x1)
        out = self.outc(x)
        if self.gt_ds:
            return (torch.sigmoid(gt_pre5),torch.sigmoid(gt_pre4),torch.sigmoid(gt_pre3),torch.sigmoid(gt_pre2),torch.sigmoid(gt_pre1)),torch.sigmoid(out)
        else:
            
            return torch.sigmoid(out) 
        

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.down5 = torch.utils.checkpoint(self.down5)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.up5 = torch.utils.checkpoint(self.up5)
        self.outc = torch.utils.checkpoint(self.outc)
