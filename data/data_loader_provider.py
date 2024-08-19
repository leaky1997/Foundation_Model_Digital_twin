from torch.utils.data import DataLoader
from .task_dataset_collection import * # import all the dataset classes
from utils.dataset_utils import init_and_merge_datasets # import the function to merge datasets
from .data_base import ClassificationDataset,\
    AnomalyDetectionDataset,\
    ImputationDataset,\
    ForecastingDataset
    

DATASET_TASK_CLASS = {
    'THU_006_Classification': ClassificationDataset,
    'THU_006_Forecasting': ForecastingDataset,
    'THU_006_Imputation': ImputationDataset,
    'THU_006_AnomalyDetection': AnomalyDetectionDataset,
    'DIRG_020_Classification': ClassificationDataset,
    'DIRG_020_Forecasting': ForecastingDataset,
    'DIRG_020_Imputation': ImputationDataset,
    'DIRG_020_AnomalyDetection': AnomalyDetectionDataset,

    
}

def get_dataloader(args, task_data_name,flag):
    
    
        
    dataset_task = DATASET_TASK_CLASS[task_data_name]
    dataset = dataset_task(args, flag=flag)
    shuffle = flag == 'train'
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers
    )
    
    if flag in ['test','val'] and 'AnomalyDetection' in task_data_name:
        dataset_train = dataset_task(args, flag='train')
        dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        dataset = [dataset_train,dataset]
        dataloader = [dataloader_train,dataloader]
    
    return dataset, dataloader

def get_data(args,task_data_config,flag = 'train'):
    dataset_list = []
    data_loader_list = []
    for iter,(task_data_name, task_config) in enumerate(task_data_config.items()):
        print(f"# {iter} loading dataset:", task_data_name)
        args.data_dir = task_config['data_dir']
        dataset, data_loader = get_dataloader(args, task_data_name, flag=flag)
        dataset_list.append(dataset)
        data_loader_list.append(data_loader)
    return dataset_list, data_loader_list


def get_train_val_test_data(args,task_data_config,flag): # TODO : add the flag to the function
    _ , train_dataloader_list = get_data(args,task_data_config,flag='train') 
    _ , val_dataloader_list = get_data(args,task_data_config,flag='val')
    _ , test_dataloader_list = get_data(args,task_data_config,flag='test')
    data_loader_cycle, train_steps = init_and_merge_datasets(train_dataloader_list)
    data_loader_cycle_val, val_steps = init_and_merge_datasets(val_dataloader_list)
    data_loader_cycle_test, test_steps = init_and_merge_datasets(test_dataloader_list)
    
    return data_loader_cycle, train_steps, data_loader_cycle_val, data_loader_cycle_test
    
    
#%%
if __name__ == '__main__':
    from data_loader_provider import get_train_val_test_data
    from utils.dataset_utils import read_task_data_config,get_task_data_config_list
    import torch
    import yaml

    from types import SimpleNamespace
    # 假设配置文件名为 config.yaml

        
    args = SimpleNamespace()
    args.task_data_config_path = '/home/user/LQ/B_Signal/Signal_foundation_model/Foundation_Model_Digital_twin/data/task_data_yaml/multi_task.yaml'
    args.batch_size = 32
    args.num_workers = 4

    task_data_configs = read_task_data_config(args.task_data_config_path)
    task_data_configs_list = get_task_data_config_list(
            task_data_configs, default_batch_size=args.batch_size)


    get_train_val_test_data(args, task_data_configs,flag='train')