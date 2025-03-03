import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix, roc_curve, f1_score, jaccard_score
from configs.config import setting_config


# @torch.inference_mode()
# def evaluate(net, dataloader, device, amp):
#     net.eval()
#     num_val_batches = len(dataloader)
#     dice_score = 0

#     # iterate over the validation set
#     with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
#         for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
#             image, mask_true = batch['image'], batch['mask']

#             # move images and labels to correct device and type
#             image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
#             mask_true = mask_true.to(device=device, dtype=torch.long)

#             # predict the mask
#             mask_pred = net(image)

#             assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
#             mask_pred = (F.sigmoid(mask_pred) > 0.2).float()
#             # compute the Dice score
            
#             dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

#     net.train()
#     return dice_score / max(num_val_batches, 1)

# 计算IoU交并比
def compute_iou(pred,label):
    intersection = np.logical_and(label,pred)
    union = np.logical_or(label,pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

@torch.inference_mode()
def eval_dice(model,dataloader,criterion,epoch,device,amp):
    model.eval()
    # num_val_batches = len(dataloader) # 224
    num_val_iterations_per_epoch = 20
    preds = []
    gts = []
    loss_list = []
    iou_scores = []
    
    # 在五折交叉验证上划分好的数据集上进行验证---->
    with torch.no_grad():
        # for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        for batch_id in range(num_val_iterations_per_epoch):
            batch = next(dataloader)
            image,mask_true = batch['data'], batch['target']   # 一张图片对应一个mask (1,3,512,512) (1,1,512,512)
            
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float, non_blocking=True)
            mask_true = mask_true.to(device=device, dtype=torch.float,non_blocking=True)
            
            # 前向传播,计算loss
            gt_pre,out = model(image)
            loss = criterion(gt_pre,out,mask_true)
            
            loss_list.append(loss.item()) # 每个批次的损失
            gts.append(mask_true.squeeze(1).cpu().detach().numpy()) # 每个批次的label-->(B,H,W),移到cpu释放内存---->去除维度()
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy() # 去除通道,转换为numpy数组
            preds.append(out) # 每次批次的网络预测
            
            # 计算和记录IoU分数
            # for pred,gt in zip(out,mask_true):
            #     iou_score = compute_iou(pred >= 0.5, gt.cpu().detach().numpy() >= 0.5)
            #     iou_scores.append(iou_score.score)
            
            loss = np.mean(loss_list)
        # 最后preds:包含224个numpy数组,每一个都是预测值  ; gts包含224个numpy数组,每一个都是label标签
            
        # 每个epoch 进行一次验证集评估dice分数 ----> 越接近越好
        # 将预测值和真实标签展平为一维数组
        preds = np.array(preds).reshape(-1) 
        gts = np.array(gts).reshape(-1) 
        
        # 将概率预测值和真实标签转换为二进制分类结果 ----> 阈值初步定为0.5
        y_pre = np.where(preds>=0.5,1,0)
        y_true = np.where(gts>=0.5,1,0)
            
        # 计算混淆矩阵
        confusion = confusion_matrix(y_true,y_pre)
        TN,FP,FN,TP = confusion[0,0],confusion[0,1],confusion[1,0],confusion[1,1]
        # 精确度
        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) !=0 else 0
        # dice分数
        dice = float(2 * TP) / float(2 * TP + FP +FN) if float(2 * TP + FP + FN) !=0 else 0
        
        fpr, tpr, thresholds = roc_curve(gts,preds)  # 注意：这里使用的是未经阈值处理的原始概率
        # 计算曲线下的面积AUC --- > 如果auc在连续几轮中没有显著提高，可能进行了过拟合--->训练可以早停，调整学习率
        roc_auc = auc(fpr, tpr)
        
        # 计算 F1-Score
        f1 = f1_score(y_true, y_pre)

        # 计算 MIOU (Mean Intersection-Over-Union)
        miou = jaccard_score(y_true, y_pre)
        
        # log_info = f'val epoch: {epoch}, loss:{np.mean(loss_list):.4f},accuracy:{accuracy},dice:{dice},AUC: {roc_auc:.4f}'
        log_info = (f'val epoch: {epoch}, loss: {loss:.4f}, accuracy: {accuracy:.4f}, '
                    f'dice: {dice:.4f}, AUC: {roc_auc:.4f}, F1: {f1:.4f}, MIOU: {miou:.4f}, ')
        print(log_info)
            
    model.train()
    return dice,roc_auc,f1,miou