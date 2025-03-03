import sys
# sys.path.append('/root/workspace1/BI_UNet/utils/')

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
from torch import Tensor
from typing import Callable


class SoftDiceLoss(nn.Module):
    def __init__(self,apply_nonline:Callable=None,batch_dice: bool=True,do_bg=False,smooth:float=1.,clip_tp: float = None):
        super(SoftDiceLoss,self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonline = apply_nonline
        self.smooth = smooth
        self.clip_tp =clip_tp
        
    def forward(self,x,y):
        # （16，1，512，512）
        if self.apply_nonline is not None:
            x = self.apply_nonline(x)
        
        # 形状全部为(2,3)
        axes = tuple(range(2,x.ndim))
        
        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0],1,*y.shape[1:])) 
            print("y:",y.shape)
            if x.shape == y.shape:
                y_onehot = y
                print("y_onehot_1",y_onehot.shape)
            else: 
                print("x:",x.shape)
                print("y:",y.long().shape)
                y_onehot = torch.zeros(x.shape,device=x.device,dtype=bool)
                print("y_onehot",y_onehot.shape)
                y_onehot.scatter_(1,y.long(),1) 
                print("y_onehot_tianchong",y_onehot.shape)
            
            if not self.do_bg:
                y_onehot = y_onehot[:,1:] 
            sum_gt = y_onehot.sum(axes) 
            
        if not self.do_bg:
            x=x[:,1:] 
        
        inserct = (x * y_onehot).sum(axes) #(32,0)
        print("inserct:",inserct.shape)
        sum_pred = x.sum(axes)
        
        if  self.batch_dice:
            inserct = inserct.sum(0)
            sum_pred =sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)
        
        dc = (2 * inserct + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth,1e-8)) 
        dc = dc.mean() # 求一个批次上的平均值
        return -dc


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0) 
        pred_ = pred.reshape(size, -1)
        target_ = target.view(size, -1) 
        intersection = pred_ * target_ 
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth) 
        dice_loss = 1 - dice_score.sum()/size 
        # print("dice_loss",dice_loss) 
        
        return dice_loss

# 二分类focal loss
class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2.5,alpha=100,reduction='mean',smooth=1e-6):
        super(BCEFocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha =alpha
        self.reduction = reduction
        self.smooth = smooth
        
    def forward(self,predict,target):
        predict = torch.clamp(predict, self.smooth, 1 - self.smooth)
        # loss = - self.alpha * (1 - predict) ** self.gamma * target * torch.log(predict) - (1 - self.alpha) * predict ** self.gamma * (1 - target) * torch.log(1 - predict)
        loss = - self.alpha * (1 - predict) ** self.gamma * target * torch.log(predict) - predict ** self.gamma * (1 - target) * torch.log(1 - predict)
        
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss 
        