import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import Trainer, seed_everything

import sys
from pathlib import Path
sys.path.insert(0,str(Path().absolute())+'/src/autoencoder2')

from concrete_autoencoder_zebra_all_series2 import ConcreteAutoencoder
from dataset2 import MRIDataModule
from logger import logger, set_log_level
from argparse2 import file_path

from nilearn.masking import apply_mask
from nilearn.masking import unmask

import numpy as np
import pandas as pd

def trainer(args: Namespace) -> None:
    """Take command line arguments to train a model.

    Args:
        args (Namespace): arguments from argparse
    """
    experiment_name = "concrete_autoencoder"

    set_log_level(args.verbose)
    is_verbose = args.verbose < 30

    logger.info("Start training with params: %s", str(args.__dict__))

    model = ConcreteAutoencoder(
        args.mridata_path,
        args.input_output_size,
        args.latent_size,
        args.latent_size2,
        args.encoder2_hidden_layers,
        learning_rate=args.learning_rate,
        max_temp=args.max_temp,
        min_temp=args.min_temp,
        reg_lambda=args.reg_lambda,
        reg_threshold=args.reg_threshold,
    )

    if args.checkpoint is not None:
        logger.info("Loading from checkpoint")
        model = model.load_from_checkpoint(
            str(args.checkpoint), hparams_file=str(args.hparams)
        )

    # We assume that the code for the subject is stored in the third column
    subjs = pd.read_csv(args.header_file,index_col=0)
    val = subjs['1']

    dm = MRIDataModule(
        subject_train = np.delete(np.unique(val),np.unique(val)==args.val_subj),
        subject_val = np.array([args.val_subj]),
        data_file=args.data_file,
        header_file=args.header_file,
        batch_size=args.batch_size,
        in_memory=args.in_memory,
    )

    plugins = []
    if args.accelerator == "ddp":
        plugins = [
            DDPPlugin(find_unused_parameters=False, gradient_as_bucket_view=True)
        ]

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            EarlyStopping(
                monitor="mean_max",
                mode="max",
                patience=float("inf"),
                stopping_threshold=args.stopping_threshold,
                verbose=is_verbose,
            ),
            ModelCheckpoint(
                monitor="mean_max",
                mode="max",
                save_top_k=1,
                verbose=is_verbose,
            ),
        ],
        checkpoint_callback=True,
        logger=TensorBoardLogger("logs", name=experiment_name),
        plugins=plugins,
    )

    if "MLFLOW_ENDPOINT_URL" in os.environ:
        mlflow.set_tracking_uri(os.environ["MLFLOW_ENDPOINT_URL"])

    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()
    mlflow.log_params(vars(args))
    
    seed_everything(args.seed_number)
    
    dm.setup("test")
    prediction = trainer.predict(model, dm.test_dataloader())
    
    pred_sig_save = torch.zeros(sum(val==args.val_subj),args.input_output_size)
    param_save = torch.zeros(sum(val==args.val_subj),args.latent_size2)
    
    param_save[param_save == 0] = float('nan')
    pred_sig_save[pred_sig_save == 0] = float('nan')
    batch_idx = 0
    for element in prediction:
        param_save[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size,:] = element[0]
        pred_sig_save[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size,:] = element[1]
        batch_idx = batch_idx+1
    
    # we estimate dperp
    dperp = param_save[:,2]*param_save[:,3]
    
    # Save as txt if there is no mask (path should be provided as txt), and as nifti if a mask is provided
    if args.mask_file:
        pred_reshape = unmask(np.transpose(pred_sig_save.cpu().detach().numpy()), args.mask_file)
        pred_reshape.to_filename(args.path_save)
        
        param_reshape = unmask(np.transpose(param_save.cpu().detach().numpy()), args.mask_file)
        param_reshape.to_filename(args.path_save_param)
        
        dperp_reshape = unmask(np.transpose(dperp.cpu().detach().numpy()), args.mask_file)
        dperp_reshape.to_filename(args.path_save_param[:-7]+'_dperp.nii.gz')
    else:
        np.savetxt(args.path_save, pred_sig_save.cpu().detach().numpy())
        np.savetxt(args.path_save_param, param_save.cpu().detach().numpy())
        np.savetxt(args.path_save_param[:-4]+'_dperp.txt', dperp.cpu().detach().numpy())

    dirs = sorted(os.listdir('logs/concrete_autoencoder'))
    command_move = 'mv logs/concrete_autoencoder/' + dirs[-1] + ' ' + args.folder_hyperparams
    os.system(command_move)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Concrete Autoencoder trainer", usage="%(prog)s [options]"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 10, 20, 30, 40, 50],
        default=20,
        metavar="XX",
        help="verbosity level (default: 10)",
    )
    """parser.add_argument(
        "--val_subj",
        default=15,
        type=int,
        metavar="N",
        help="subject employed for validation (default: 15)",
    )"""
    parser.add_argument(
        "--seed_number",
        default=42,
        type=int,
        metavar="N",
       help="seed employed to initialise the job (default: 42)",
    )
    parser.add_argument(
        "--folder_hyperparams",
        default=None,
        type=str,
        required=False,
        metavar="PATH",
        help="path to save the folder with hyperparameters and checkpoints",
    )
    parser.add_argument(
        "--path_save",
        type=str,
        required=False,
        metavar="PATH",
        help="file name of the path to save the predicted signal",
    )
    parser.add_argument(
        "--path_save_param",
        type=str,
        required=False,
        metavar="PATH",
        help="file name of the path to save the predicted maps of parameters",
    )
    parser.add_argument(
        "--mask_file",
        type=str,
        required=False,
        metavar="PATH",
        help="file name of the path with the mask",
    )
    parser.add_argument(
        "--stopping_threshold",
        default=0.998,
        type=float,
        required=False,
        metavar="N",
        help="Mean max value used as stopping threshold value before reaching the established maximum number of epochs",
    )

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = ConcreteAutoencoder.add_model_specific_args(parent_parser=parser)
    parser = MRIDataModule.add_model_specific_args(parent_parser=parser)

    args = parser.parse_args()
    trainer(args)
