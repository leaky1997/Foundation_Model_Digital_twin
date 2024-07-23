import yaml
import os
from types import SimpleNamespace
import time

def parse_arguments(parser):
    # TODO
    # 解析参数
    
    args_dir = parser.parse_args()
    # 使用参数
    trainer_config_dir = args_dir.trainer_config_dir
    # 读取YAML文件
    with open(trainer_config_dir, 'r') as f:
        trainer_config = yaml.safe_load(f)

    task_data_config_dir = args_dir.task_data_config_dir
    # 读取YAML文件
    with open(task_data_config_dir, 'r') as f:
        task_data_config = yaml.safe_load(f)
    
    # dataset = args.data_dir[-3:].replace('/','')
    time_stamp = time.strftime("%d-%H-%M-%S", time.localtime())
    # name = f'model_{trainer_config_dir.model}time{time_stamp}_lr{trainer_config_dir.learning_rate}_epochs{trainer_config_dir.num_epochs}_dataset{task_data_config_dir.dataset_task}'

    print(f'Running experiment: {name}')
    
    # if args.debug != 'True':
    path = 'save/' + f'task_{args.dataset_task}/'+f'model_{args.model}/' + name
    if not os.path.exists(path):
        os.makedirs(path)
    trainer_config.path = path
    return trainer_config,task_data_config,path