import torch
import torch.nn as nn
import pytorch_lightning as pl
import importlib
from pytorch_lightning import seed_everything

from utils.dataset_utils import read_task_data_config, get_task_data_config_list
from utils.exp_utils import custom_print_decorator,\
    scheduler_dict,get_loss_by_name,apply_random_mask_for_imputation
from torch import optim

print = custom_print_decorator(print)


class Exp(pl.LightningModule):
    def __init__(self,args):
        super(Exp, self).__init__()
        self.args = args
        seed_everything(args.seed)
        self.path = self.args.path
        
        args_dict = vars(args)
        self.save_hyperparameters(args_dict)
        
        self.task_data_config = read_task_data_config(self.args.task_data_config_path)
        self.task_data_config_list= get_task_data_config_list(
            self.task_data_config, default_batch_size=self.args.batch_size)
        
        self.model = self.init_model(args.model)

        self.criterion_list = self.init_loss(self.task_data_config_list) # TODO check
        self.init_task_to_method()
        
    def init_model(self, model):
        module = importlib.import_module("model."+self.args.model) # 
        model = module.Model(
                    self.args, self.task_data_config_list)     
        return model    

    
    def init_loss(self, config_list):  # sup
        criterion_list = []
        for each_config in config_list:
            if 'loss' in each_config[1]:
                loss_name = each_config[1]['loss']
            else:
                if each_config[1]['task_name'] == 'Forecasting':
                    loss_name = 'MSE'
                elif each_config[1]['task_name'] == 'Classification':
                    loss_name = 'CE'
                elif each_config[1]['task_name'] == 'Imputation':
                    loss_name = 'MSE'
                elif each_config[1]['task_name'] == 'Anomaly_detection':
                    loss_name = 'MSE'
                else:
                    print("this task has no loss now!", folder=self.path)
                    exit()
            criterion_list.append(get_loss_by_name(loss_name))

        return criterion_list
    
    def init_metric(self, config_list): # TODO
        pass
    
    def choose_training_parts(self, prompt_tune=False):
        for name, param in self.model.named_parameters():
            if prompt_tune:
                if ('prompt_token' in name 
                    or 'mask_prompt' in name 
                    or 'cls_prompt' in name 
                    or 'mask_token' in name 
                    or 'cls_token' in name 
                    or 'category_token' in name):
                    
                    param.requires_grad = True
                    print("trainable:", name)
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

        if not prompt_tune:
            print("all trainable.")


    def init_task_to_method(self):
        
        self.task_to_method = {
        'Forecasting': self.train_long_term_forecast,
        'Classification': self.train_classification,
        'Imputation': self.train_imputation,
        'Anomaly_detection': self.train_anomaly_detection}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        
        sample,task_id = batch
        
        task_name = self.task_data_config_list[task_id][1]['task_name']
        if task_name in self.task_to_method:
            loss = self.task_to_method[task_name](
                self.model,
                sample,
                self.criterion_list[task_id],
                self.task_data_config_list[task_id][1],
                task_id)
            self.log(f'train_{task_name}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        else:
            raise ValueError(f"Unknown task name: {task_name}")
        return loss
    
    def validation_step(self, batch, batch_idx):
        sample,task_id = batch
        
        task_name = self.task_data_config_list[task_id][1]['task_name']
        if task_name in self.task_to_method:
            loss = self.task_to_method[task_name](
                self.model,
                sample,
                self.criterion_list[task_id],
                self.task_data_config_list[task_id][1],
                task_id)
            self.log(f'val_{task_name}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        else:
            raise ValueError(f"Unknown task name: {task_name}")
        return loss
    
    def test_step(self, batch, batch_idx):
        sample,task_id = batch
        
        task_name = self.task_data_config_list[task_id][1]['task_name']
        if task_name in self.task_to_method:
            loss = self.task_to_method[task_name](
                self.model,
                sample,
                self.criterion_list[task_id],
                self.task_data_config_list[task_id][1],
                task_id)
            self.log(f'test_{task_name}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        else:
            raise ValueError(f"Unknown task name: {task_name}")
        return loss
    
    # def train_one_batch(self, batch):
    def train_long_term_forecast(self, model, this_batch, criterion, config, task_id):
        label_len = config['label_len']
        pred_len = config['pred_len']
        task_name = config['task_name']
        features = config['features']

        batch_x, batch_y, _, _ = this_batch

        batch_x = batch_x.float().to(self.device_id)
        batch_y = batch_y.float().to(self.device_id)

        dec_inp = None
        dec_inp = None
        batch_x_mark = None
        batch_y_mark = None

        with torch.cuda.amp.autocast():
            outputs = model(batch_x, batch_x_mark, dec_inp,
                            batch_y_mark, task_id=task_id, task_name=task_name)
            f_dim = -1 if features == 'MS' else 0
            outputs = outputs[:, -pred_len:, f_dim:] # B L C
            batch_y = batch_y[:, -pred_len:, f_dim:]
            loss = criterion(outputs, batch_y)

        return loss

    def train_classification(self, model, this_batch, criterion, config, task_id):
        task_name = config['task_name']

        batch_x, label = this_batch

        batch_x = batch_x.float().to(self.device_id)
        padding_mask = padding_mask.float().to(self.device_id)
        label = label.to(self.device_id)
        with torch.cuda.amp.autocast():
            outputs = model(batch_x,  task_id=task_id, task_name=task_name)
            if outputs.shape[0] == label.shape[0]:
                loss = criterion(outputs, label.long().squeeze(-1)) # CE B,C VS B,1
            else:
                label = label.repeat(outputs.shape[0]//label.shape[0], 1)
                loss = criterion(outputs, label.long().squeeze(-1))

        return loss

    def train_imputation(self, model, this_batch, criterion, config, task_id):
        task_name = config['task_name']
        # features = config['features']

        batch_x, _, _, _ = this_batch
        batch_x = batch_x.float().to(self.device_id)

        # block-wise imputation
        inp, mask = apply_random_mask_for_imputation(
            batch_x, self.args.patch_len, self.args.mask_rate)

        with torch.cuda.amp.autocast():
            outputs = model(inp, None, None,
                            None, task_id=task_id, mask=mask, task_name=task_name)
        # f_dim = -1 if features == 'MS' else 0
        # outputs = outputs[:, :, f_dim:]
        loss = criterion(outputs[mask == 0], batch_x[mask == 0])

        return loss

    def train_anomaly_detection(self, model, this_batch, criterion, config, task_id): # TODO
        task_name = config['task_name']
        # features = config['features']

        batch_x, _ = this_batch

        batch_x = batch_x.float().to(self.device_id)

        with torch.cuda.amp.autocast():
            outputs = model(batch_x, None, None,
                            None, task_id=task_id, task_name=task_name)
            # f_dim = -1 if features == 'MS' else 0
            # outputs = outputs[:, :, f_dim:]
            loss = criterion(outputs, batch_x)

        return loss

    
    # def testing_step(self, batch, batch_idx):
        

    
    
    def configure_optimizers(self): # 'ReduceLROnPlateau'
        '''定义模型优化器'''
        optimizer = optim.Adam(self.model.parameters(),
                         lr=self.args.learning_rate,
                         weight_decay=self.args.weight_decay)
        scheduler = scheduler_dict[self.args.scheduler]
        out = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler(optimizer), # 'ReduceLROnPlateau'
                "monitor": self.args.monitor,
                "frequency": self.args.patience
            },
        }
        self.optimizer = optimizer
        return out
    
    def load_best_model(self):
        pass