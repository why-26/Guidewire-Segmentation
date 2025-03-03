import torch
from torch import Tensor, nn
from .dice_score import DiceLoss,BCEFocalLoss
from .cldice import soft_dice_cldice
from .tcloss import CombinedLoss

class DC_and_BCE_loss(nn.Module):
    def __init__(self,bce_kwargs,soft_dice_kwargs,weight_ce=0.5,weight_dice=1,dice_class=soft_dice_cldice):
        super(DC_and_BCE_loss,self).__init__()
        
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ce = nn.BCEWithLogitsLoss()
        self.dc = dice_class(**soft_dice_kwargs)
        
    def forward(self,net_output:torch.Tensor,target:torch.Tensor):
        dc_loss = self.dc(net_output,target) 
        ce_loss = self.ce(net_output,target) 
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss 
        return result

class Supervision_Loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs):
        super(Supervision_Loss,self).__init__()
        self.bcedice = DC_and_BCE_loss(bce_kwargs, soft_dice_kwargs, dice_class=soft_dice_cldice)

        
    def forward(self, gt_pre,out,target):
        bcediceloss = self.bcedice(out,target)
        # weights = [0.1,0.2,0.3,0.4,0.5]
        weights = [0.03125,0.0625,0.125,0.25,0.5]
        loss = 0
        gt_loss = 0
        for i in range(len(gt_pre)): 
            size = target.size(0)
            gt = gt_pre[i] 
            loss = weights[i] * self.bcedice(gt,target) 
            gt_loss += loss 
        return bcediceloss + gt_loss 
    
    
    

        
        
