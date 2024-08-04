from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer    #  lighting.pytorch? check if bug
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

def get_trainer(args,path):

    callback_list = call_backs(args,path)
    log_list = [CSVLogger(path, name="logs"),
                WandbLogger(project=args.dataset_task)] if args.wandb_flag else [CSVLogger(path, name="logs")]

    trainer = pl.Trainer(callbacks=callback_list,
                        max_epochs=args.num_epochs,
                        devices= args.gpus,
                        logger = log_list,
                        log_every_n_steps=1,
                        precision=16,)

def call_backs(args,path):
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='checkpoint_{epoch:02d}',
        save_top_k=8,
        save_last=True,
        mode='min',
        dirpath = path
    )
    # 初始化训练器
    callback_list = [checkpoint_callback]
    
    early_stopping = create_early_stopping_callback(args)
    callback_list.append(early_stopping)
    
    return callback_list

def latent_obseaver_call_back():
    '''
    提取隐层特征并保存
    '''
    pass

def create_early_stopping_callback(args):
    """
    根据args参数创建EarlyStopping回调实例。
    
    参数:
    - args: 包含配置的对象，比如Namespace对象。
    
    返回:
    - 一个配置好的EarlyStopping实例。
    
    """
        # 使用args中指定的patience值
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=args.patience,  # 从args中读取patience值
        verbose=True,
        mode='min',
        check_finite=True,  # 当监控指标为无穷大或NaN时停止
        check_on_train_epoch_end=False  # 仅在验证阶段检查
    )
    
    return early_stopping