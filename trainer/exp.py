import torch
import torch.nn as nn
import pytorch_lightning as pl
import importlib
from pytorch_lightning import seed_everything

from utils.dataset_utils import read_task_data_config, get_task_data_config_list
from utils.exp_utils import custom_print_decorator


print = custom_print_decorator(print)

class Exp(pl.LightningModule):
    def __init__(self,args):
        super(Exp, self).__init__()
        self.args = args
        seed_everything(args.seed)
        
        args_dict = vars(args)
        self.save_hyperparameters(args_dict)
        
        self.model = self.init_model(args.model)

    def init_model(self, model):
        module = importlib.import_module("models."+self.args.model) # 
        model = module.Model(
                    self.args, self.task_data_config_list, pretrain=True)     
        return model    
    
    # def init_task_data(self, flag):
    #     data_set_list = []
    #     data_loader_list = []
    #     for task_data_name, task_config in self.task_data_config.items():
    #         print("loading dataset:", task_data_name, folder=self.path)
    #         data_set, data_loader = data_provider(
    #             self.args, task_config, flag, ddp=True)
    #         data_set_list.append(data_set)
    #         data_loader_list.append(data_loader)
    #     return data_set_list, data_loader_list
    
    def init_loss(self, loss):
        self.loss = loss
        pass
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layer(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss
        pass
    
    def train_one_batch(self, batch):
        pass
    
    def configure_optimizers(self):
        pass
    
    def load_best_model(self):
        pass