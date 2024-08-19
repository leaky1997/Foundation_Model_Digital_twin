import torch
import torch.nn as nn
import pytorch_lightning as pl
import importlib
from pytorch_lightning import seed_everything

from utils.dataset_utils import read_task_data_config, get_task_data_config_list
from utils.exp_utils import custom_print_decorator,\
    scheduler_dict,get_loss_by_name,apply_random_mask_for_imputation,split_batch,resample,\
        metric,cal_accuracy
from torch import optim
import torchmetrics

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
        model = module.Model(self.args, self.task_data_config_list)     
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
        self._metric = metric
        self.acc = cal_accuracy # TODO
    
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
        
        self.test_task_to_method = {
        'Forecasting': self.train_long_term_forecast,
        'Classification': self.train_classification,
        'Imputation': self.train_imputation,
        'Anomaly_detection': self.test_anomaly_detection}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx,flag = 'train'):
        
        sample,task_id = batch
        if self.args.resample:
            sample = resample(sample, self.args.resample[0], self.args.resample[1])
        
        
        task_name = self.task_data_config_list[task_id][1]['task_name']
        if task_name in self.task_to_method:
            loss = self.task_to_method[task_name](
                self.model,
                sample,
                self.criterion_list[task_id],
                self.task_data_config_list[task_id][1],
                task_id,
                flag = flag)
            self.log(f'{flag}_{task_name}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        else:
            raise ValueError(f"Unknown task name: {task_name}")
        return loss
    
    def validation_step(self, batch, batch_idx,flag = 'val'):
        sample,task_id = batch
        
        task_name = self.task_data_config_list[task_id][1]['task_name']
        if task_name in self.task_to_method:
            loss = self.test_task_to_method[task_name](
                self.model,
                sample,
                self.criterion_list[task_id],
                self.task_data_config_list[task_id][1],
                task_id,
                flag = flag)
            self.log(f'{flag}_{task_name}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        else:
            raise ValueError(f"Unknown task name: {task_name}")
        return loss
    
    def test_step(self, batch, batch_idx,flag = 'test'):
        sample,task_id = batch
        
        task_name = self.task_data_config_list[task_id][1]['task_name']
        if task_name in self.task_to_method:
            loss = self.task_to_method[task_name](
                self.model,
                sample,
                self.criterion_list[task_id],
                self.task_data_config_list[task_id][1],
                task_id,
                flag = flag)
            self.log(f'{flag}_{task_name}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        else:
            raise ValueError(f"Unknown task name: {task_name}")
        return loss
    
    # def train_one_batch(self, batch):
    def train_long_term_forecast(self, model, this_batch, criterion, config, task_id,flag = 'train'):
        label_len = config['label_len']
        pred_len = config['pred_len']
        task_name = config['task_name']


        batch_x, _  = this_batch
        batch_x, batch_y = split_batch(batch_x, label_len, pred_len)
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        with torch.cuda.amp.autocast():
            outputs = model(batch_x, task_id=task_id, task_name=task_name)
            # f_dim = -1 if features == 'MS' else 0
            # outputs = outputs[:, -pred_len:] # B L C
            # batch_y = batch_y[:, -pred_len:]
            loss = criterion(outputs, batch_y)

        return loss

    def train_classification(self, model, this_batch, criterion, config, task_id,flag = 'train'):
        task_name = config['task_name']

        batch_x, label = this_batch

        batch_x = batch_x.float()
        # padding_mask = padding_mask.float()
        label = label
        with torch.cuda.amp.autocast():
            outputs = model(batch_x,  task_id=task_id, task_name=task_name)
            # if outputs.shape[0] == label.shape[0]:
            loss = criterion(outputs, label.long().squeeze(-1)) # CE B,C VS B,1
            # else:
            #     label = label.repeat(outputs.shape[0]//label.shape[0], 1)
            #     loss = criterion(outputs, label.long().squeeze(-1))
            outputs = torch.argmax(outputs, dim=-1)
            acc = self.acc(outputs, label.long().squeeze(-1))
            self.log(f'{flag}_{task_name}_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)

        return loss

    def train_imputation(self, model, this_batch, criterion, config, task_id,flag = 'train'):
        task_name = config['task_name']
        # features = config['features']

        batch_x, _  = this_batch
        batch_x = batch_x.float()

        # block-wise imputation
        inp, mask = apply_random_mask_for_imputation(
            batch_x, self.args.patch_len, self.args.mask_rate)

        with torch.cuda.amp.autocast():
            outputs = model(inp, task_id=task_id, mask=mask, task_name=task_name)
        # f_dim = -1 if features == 'MS' else 0
        # outputs = outputs[:, :, f_dim:]
        loss = criterion(outputs[mask == 0], batch_x[mask == 0])

        return loss

    def train_anomaly_detection(self, model, this_batch, criterion, config, task_id,flag = 'train'): # TODO
        task_name = config['task_name']
        # features = config['features']

        batch_x, _ = this_batch

        batch_x = batch_x.float()

        with torch.cuda.amp.autocast():
            outputs = model(batch_x, task_id=task_id, task_name=task_name)
            # f_dim = -1 if features == 'MS' else 0
            # outputs = outputs[:, :, f_dim:]
            loss = criterion(outputs, batch_x)

        return loss

    
#%% test
        

    def test_anomaly_detection(self, test_loader_set, data_task_name, task_id):
        train_loader, test_loader = test_loader_set
        
        anomaly_criterion = nn.MSELoss(reduce=False)
        
        
        attens_energy = []
        self.model.eval()
        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float()
                # reconstruction
                outputs = self.model(
                    batch_x, task_id=task_id, task_name='anomaly_detection')
                # criterion
                score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1) # TODO
                score = score.detach().cpu()
                attens_energy.append(score)
                
                train_energy = np.concatenate(attens_energy, axis=0)

        # (2) stastic on the test set
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float()
                # reconstruction
                outputs = self.model(batch_x, task_id=task_id, task_name='anomaly_detection')
                # criterion
                score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu()
                attens_energy.append(score)
                test_labels.append(batch_y)

                test_energy = np.concatenate(attens_energy, axis=0)
        
        # (2.5) find the threshold
        combined_energy = np.concatenate([train_energy, test_energy], axis=0).reshape(-1)
        threshold = np.percentile(
            combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set ############
            # 计算每个样本中超过 threshold 的元素比例
        rate = np.mean(test_energy > threshold, axis=1)

        # 判断这些比例是否超过 signal_threshold
        pred = (rate > self.args.signal_threshold).astype(int)
        # pred = (test_energy > threshold).astype(int)  # 统计 rate = >point_threshold的百分比，如果 rate > signal_threshold(50%) 则 pred 为1
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        # gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        return f_score
    
    
    def test_classification(self, test_loader_set, data_task_name, task_id):
        
        task_name = config['task_name']

        batch_x, label = this_batch

        batch_x = batch_x.float()
        # padding_mask = padding_mask.float()
        label = label
        with torch.cuda.amp.autocast():
            outputs = model(batch_x,  task_id=task_id, task_name=task_name)
            # if outputs.shape[0] == label.shape[0]:
            loss = criterion(outputs, label.long().squeeze(-1)) # CE B,C VS B,1
            # else:
            #     label = label.repeat(outputs.shape[0]//label.shape[0], 1)
            #     loss = criterion(outputs, label.long().squeeze(-1))
            outputs = torch.argmax(outputs, dim=-1)
            acc = self.acc(outputs, label.long().squeeze(-1))
            self.log(f'{flag}_{task_name}_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
    
    def test_forecasting(self, test_loader_set, data_task_name, task_id):
            
        pass
    
    def test_imputation(self, test_loader_set, data_task_name, task_id):
        pass
    
    
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
        # 假设你已经在训练过程中使用了 ModelCheckpoint 回调
        checkpoint_callback = [cb for cb in self.trainer.callbacks if isinstance(cb, pl.callbacks.ModelCheckpoint)][0]
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            self.load_from_checkpoint(best_model_path)
        else:
            print("No best model found. Please ensure that ModelCheckpoint callback is used during training.")