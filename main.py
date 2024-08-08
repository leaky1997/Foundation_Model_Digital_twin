import argparse
import torch
from utils.config_utils import parse_arguments
from trainer.exp import Exp
from trainer.trainer_config import get_trainer

from data.data_loader_provider import get_data,get_train_val_test_data

parser = argparse.ArgumentParser(description='FMDT')

# 添加参数
parser.add_argument('--trainer_config_dir', type=str, default='config/dummy_config.yaml',
                    help='The directory of the configuration file of trainer')
parser.add_argument('--task_data_config_dir', type=str, default='data/task_data_yaml/multi_task.yaml',
                    help='The directory of the configuration file of task data')
# parser.add_argument('--check_point_dir', type=str, default='False',
#                     help='set check_point if you have checkpoint to continue training')

args, task_data_configs = parse_arguments(parser)

# config exp
exp = Exp(args)
trainer = get_trainer(args)

train_dataloader_list, train_steps, val_dataloader_list,\
    test_dataloader_list = get_train_val_test_data(args,exp.task_data_config,flag='train')

# exp start
trainer.fit(exp,
            train_dataloader_list,
            val_dataloader_list,
            # ckpt_path = args.check_point_dir # TODO when have checkpoint
            )
exp.load_best_model()
result = trainer.test(exp,test_dataloader_list)