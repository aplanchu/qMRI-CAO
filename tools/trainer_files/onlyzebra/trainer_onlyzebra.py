import os
from argparse import ArgumentParser, Namespace

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import Trainer, seed_everything

import sys
from pathlib import Path
sys.path.insert(0,str(Path().absolute())+'/src/autoencoder2')

from concrete_autoencoder_zebraorig import ConcreteAutoencoder
from dataset2 import MRIDataModule
from logger import logger, set_log_level
from argparse2 import file_path

import numpy as np
import torch
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
        args.ind_path,
        args.mridata_path,
        args.latent_size,
        args.latent_size2,
        args.encoder2_hidden_layers,
        learning_rate=args.learning_rate,
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
                monitor="val_loss",
                mode="min",
                patience=float("inf"),
                stopping_threshold=args.stopping_threshold,
                verbose=is_verbose,
            ),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
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

    trainer.fit(model, dm)
    
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
        "--stopping_threshold",
        default=1e-5,
        type=float,
        required=False,
        metavar="N",
        help="Loss value (mean-squared error) used as stopping threshold value before reaching the established maximum number of epochs",
    )

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = ConcreteAutoencoder.add_model_specific_args(parent_parser=parser)
    parser = MRIDataModule.add_model_specific_args(parent_parser=parser)

    args = parser.parse_args()
    trainer(args)
