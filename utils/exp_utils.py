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
def check_attr(args,attr = 'attention_norm'):
    if not hasattr(args, attr):
        setattr(args, attr, False)
        
def custom_print_decorator(func):
    def wrapper(*args, **kwargs):
        text = ' '.join(map(str, args))
        if 'file' not in kwargs or kwargs['file'] is None:
            sys.stdout.write(text + '\n')
        else:
            kwargs['file'].write(text + '\n')

        if 'folder' in kwargs and kwargs['folder']:
            with open(f'{kwargs["folder"]}/finetune_output.log', 'a') as log_file:
                log_file.write(text + '\n')
        if 'folder' in kwargs:
            del kwargs['folder']
        if 'file' in kwargs:
            del kwargs['file']
    return wrapper