from torch.utils.data import DataLoader
from task_dataset_collection import * # import all the dataset classes
from utils.dataset_utils import init_and_merge_datasets # import the function to merge datasets

DATASET_TASK_CLASS = {
    'THU_006_basic': THU_006or018_basic,
    'THU_018_basic': THU_006or018_basic,
    'THU_018_few_shot': THU_006or018_few_shot,
    'THU_006_few_shot': THU_006or018_few_shot,
    'THU_006_generalization': THU_006_generalization,
    'DIRG_020_basic': DIRG_020_basic,
    'DIRG_020_geberalization': DIRG_020_generalization
    
}

def get_dataloader(args, task_config,flag):
    dataset_task_config = task_config['dataset_task']
    dataset_task = DATASET_TASK_CLASS[dataset_task_config]
    dataset = dataset_task(args, flag=flag)
    shuffle = flag == 'train'
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers
    )
    return dataset, dataloader

def get_data(args,task_data_config,flag = 'train'):
    dataset_list = []
    data_loader_list = []
    for task_data_name, task_config in task_data_config.items():
        print("# loading dataset:", task_data_name)
        dataset, data_loader = get_dataloader(args, task_config, flag=flag)
        dataset_list.append(dataset)
        data_loader_list.append(data_loader)
    return dataset_list, data_loader_list


def get_train_val_test_data(args,flag): # TODO : add the flag to the function
    _ , train_dataloader_list = get_data(args,flag='train') 
    _ , val_dataloader_list = get_data(args,flag='val')
    _ , test_dataloader_list = get_data(args,flag='test')
    data_loader_cycle, train_steps = init_and_merge_datasets(train_dataloader_list)
    return data_loader_cycle, train_steps, val_dataloader_list, test_dataloader_list
    
    
    
# def get_dataset_signal(args):
#     dataset_class = DATASET_TASK_CLASS[args.dataset_task]
    
#     dataset = dataset_class(args,flag = 'train')
#     train_loader = DataLoader(
#         dataset = dataset,
#         batch_size= args.batch_size,
#         shuffle = True,
#         num_workers = args.num_workers
#     )
#     dataset = dataset_class(args,flag = 'val')
#     val_loader = DataLoader(
#         dataset = dataset,
#         batch_size= args.batch_size,
#         shuffle = False,
#         num_workers = args.num_workers
#     )
#     dataset = dataset_class(args,flag = 'test')
#     test_loader = DataLoader(
#         dataset = dataset,
#         batch_size= args.batch_size,
#         shuffle = False,
#         num_workers = args.num_workers
#     )     
#     return train_loader,val_loader,test_loader