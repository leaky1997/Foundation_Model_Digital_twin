import argparse
import torch
from utils.config_utils import parse_arguments
from trainer import Exp
from trainer.trainer_config import get_trainer

from data.data_loader_provider import get_data

parser = argparse.ArgumentParser(description='TSPN')

# 添加参数
parser.add_argument('--trainer_config_dir', type=str, default='configs/config_basic.yaml',
                    help='The directory of the configuration file of trainer')
parser.add_argument('--task_data_config_dir', type=str, default='configs/config_basic.yaml',
                    help='The directory of the configuration file of task data')

configs,args,path = parse_arguments(parser)

exp = Exp(args)
trainer = get_trainer(args)
train_dataloader, val_dataloader, test_dataloader = get_data(args)

trainer.fit(exp, train_dataloader, val_dataloader)
exp.load_best_model()
result = trainer.test(exp,test_dataloader)