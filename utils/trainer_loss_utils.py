import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import F1Score



# 1 mixup_batch
def mixup_batch(batch,alpha = 0.8):
    # mix_ratio = np.random.dirichlet(np.ones(3) * 0.9,size=1) # 设置为0.9
    lamda = np.random.beta(alpha,alpha)
    index = torch.randperm(batch['split_data'].size(0)).cuda()
    
    mixed_batch = {}
    for k, v in batch.items():
        mixed_batch[k] = lamda*v + (1-lamda)*v[index]
        
    return batch

# 2 mixuploss
def mixup(x,y,criterion,net,alpha = 0.8):
    # mix_ratio = np.random.dirichlet(np.ones(3) * 0.9,size=1) # 设置为0.9
    lamda = np.random.beta(alpha,alpha)
    index = torch.randperm(x.size(0)).cuda()
    x = lamda*x + (1-lamda)*x[index,:]
    y = lamda*y + (1-lamda)*y[index,:]
    y_pre = net(x)
    loss = criterion(y_pre,torch.max(y, 1)[1])
    return loss

# 