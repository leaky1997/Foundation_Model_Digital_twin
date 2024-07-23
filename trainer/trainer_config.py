from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer    #  lighting.pytorch? check if bug
from pytorch_lightning.loggers import WandbLogger


def get_trainer(args):
    wandb_logger = WandbLogger(project=args.dataset_task)

    callback_list = call_backs(args,path)

    if not hasattr(args, 'wandb_flag'):
        setattr(args, 'wandb_flag', True)  # TODO add this arg to fix bug
    log_list = [CSVLogger(path, name="logs"),wandb_logger] if args.wandb_flag else [CSVLogger(path, name="logs")]


    trainer = pl.Trainer(callbacks=callback_list,
                        max_epochs=args.num_epochs,
                        devices= args.gpus,
                        logger = log_list,# ,TensorBoardLogger(path, name="logs")],
                        log_every_n_steps=1,)