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
parser.add_argument('--check_point_dir', type=str, default='False',
                    help='set check_point if you have checkpoint to continue training')   

configs,args,path = parse_arguments(parser)

# config exp
exp = Exp(args)
trainer = get_trainer(args)

train_dataloader_list, train_steps, val_dataloader_list,\
    test_dataloader_list = get_data(args,configs['task_data'],flag='train')



# exp start
trainer.fit(exp,
            train_dataloader_list,
            val_dataloader_list,
            ckpt_path = args.check_point_dir
            )
exp.load_best_model()
result = trainer.test(exp,test_dataloader_list)