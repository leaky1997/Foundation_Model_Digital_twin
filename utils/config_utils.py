import yaml
import os
from types import SimpleNamespace
import time

def parse_arguments(parser):
    # 解析参数
    args_dir = parser.parse_args()
    
    # 读取 trainer 配置文件
    trainer_config_dir = args_dir.trainer_config_dir
    with open(trainer_config_dir, 'r') as f:
        trainer_config = yaml.safe_load(f)
    trainer_config = SimpleNamespace(**trainer_config)

    # 读取 task 数据配置文件
    task_data_config_dir = args_dir.task_data_config_dir
    with open(task_data_config_dir, 'r') as f:
        task_data_config = yaml.safe_load(f)
    task_data_config = SimpleNamespace(**task_data_config)
    
    trainer_config.task_data_config_path = task_data_config_dir
    
    # 生成时间戳
    time_stamp = time.strftime("%d_%H_%M", time.localtime())

    # 生成保存路径
    path = os.path.join(
        trainer_config.save_dir,
        f'task_{len(task_data_config.task_dataset)}',
        f'model_{trainer_config.model}',
        time_stamp
    )
    os.makedirs(path, exist_ok=True)
    trainer_config.path = path

    # 将参数保存到路径中
    trainer_config_path = os.path.join(path, 'trainer_config.yaml')
    task_data_config_path = os.path.join(path, 'task_data_config.yaml')
    
    with open(trainer_config_path, 'w') as f:
        yaml.dump(trainer_config.__dict__, f)
    
    with open(task_data_config_path, 'w') as f:
        yaml.dump(task_data_config.__dict__, f)

    return trainer_config, task_data_config