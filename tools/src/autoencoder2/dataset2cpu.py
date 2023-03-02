import itertools
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import psutil
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd

import sys
import pathlib
sys.path.insert(0,str(Path().absolute()))

from argparse2 import file_path
from logger import logger
from autoencoder.spherical.harmonics import (
    convert_cart_to_s2,
    gram_schmidt_sh_inv,
    sh_basis_real,
)


class MRIMemorySHDataset(Dataset):
    def __init__(
        self,
        data_file_path: Path,
        subject_list: list[int],
        exclude: Optional[list[int]] = None,
        include: Optional[list[int]] = None,
        l_bandwidth: list[int] = [0, 0, 0, 0, 2],
        symmetric: bool = True,
        gram_schmidt_n_iters: int = 1000,
    ):
        """Create a dataset from the selected subjects in the subject list with matching spherical harmonics.

        Args:
            data_file_path (Path): Data h5 file path.
            subject_list (list[int]): ist of all the subjects to include.
        """
        self._data_file_path = data_file_path
        self._subject_list = subject_list
        self._l_bandwidth = l_bandwidth
        self._l_max = np.max(self._l_bandwidth)
        self._symmetric = 2 if symmetric else 1
        self._gram_schmidt_n_iters = gram_schmidt_n_iters

        assert (
            exclude is None or include is None
        ), "Only specify include or exclude, not both."

        # load the data in memory. The total file is *only* 3.1GB so it should be
        # doable on most systems. Lets check anyway...
        file_size = os.path.getsize(data_file_path)
        available_memory = psutil.virtual_memory().available

        assert (
            available_memory - file_size >= 0
        ), f"Data file requires {file_size:,} bytes of memory but {available_memory:,} was available"

        with h5py.File(data_file_path, "r") as archive:
            scheme = archive.get("scheme")[()]

            assert np.unique(scheme[:, 3]).shape != len(
                self._l_bandwidth
            ), "Length of l_bandwidth should be equal to the number of unique b values."

            indexes = archive.get("index")[()]
            # indexes of the data we want to load
            (selection, *_) = np.where(np.isin(indexes, subject_list))

            data = archive.get("data1")[selection]

            if include is not None:
                scheme = scheme[include]
                data = data[:, include]
            elif exclude is not None:
                scheme = np.delete(scheme, exclude, axis=0)
                data = np.delete(data, exclude, axis=1)

            self.sh_coefficients = self._load_sh_coefficients(archive, data, scheme)

    def _load_sh_coefficients(
        self, archive: h5py.File, data, scheme
    ) -> list[np.ndarray]:
        b_s = np.unique(scheme[:, 3])  # 5 unique values
        ti_s = np.unique(scheme[:, 4])  # 28 unique values
        te_s = np.unique(scheme[:, 5])  # 3 unique values

        ti_n = ti_s.shape[0]
        te_n = te_s.shape[0]
        b_n = b_s.shape[0]

        prev_b = b_s[0]
        sh_coefficients_b_idx = {0: 0, 2: 0}
        sh_coefficients = {
            0: torch.empty((ti_n, te_n, data.shape[0], b_n, 1)),
            2: torch.empty((ti_n, te_n, data.shape[0], 1, 5)),
        }
        for (ti_idx, ti), (te_idx, te), (b_idx, b) in itertools.product(
            enumerate(ti_s),
            enumerate(te_s),
            enumerate(b_s),
        ):
            l = self._l_bandwidth[b_idx]

            # If we've visited all b values, we reset the counter
            if prev_b == b:
                sh_coefficients_b_idx = {0: 0, 2: 0}
                prev_b = b

            filter_scheme = (
                (scheme[:, 3] == b) & (scheme[:, 4] == ti) & (scheme[:, 5] == te)
            )

            data_filtered = data[:, filter_scheme]
            if not data_filtered.any():
                continue
            data_filtered = torch.from_numpy(data_filtered).unsqueeze(2)

            gradients_xyz = scheme[filter_scheme][:, :3]
            gradients_s2 = convert_cart_to_s2(gradients_xyz)

            y = sh_basis_real(gradients_s2, l)
            y_inv = gram_schmidt_sh_inv(y, l, n_iters=self._gram_schmidt_n_iters)
            y_inv = y_inv[np.newaxis, :, :]
            y_inv = torch.from_numpy(y_inv)

            sh_coefficient = torch.einsum("npc,clp->ncl", data_filtered, y_inv)

            # extract even covariants
            s = 0
            for l in range(0, l + 1, self._symmetric):
                o = 2 * l + 1

                sh_coefficients[l][
                    ti_idx, te_idx, :, sh_coefficients_b_idx[l]
                ] = sh_coefficient[:, 0, torch.arange(s, s + o)]

                s += o
                sh_coefficients_b_idx[l] += 1

        return sh_coefficients

    def __len__(self):
        """Denotes the total number of samples"""
        return self.sh_coefficients[0].shape[2]

    def __getitem__(self, index):
        """Generates one sample of data"""
        return {k: v[:, :, index] for (k, v) in self.sh_coefficients.items()}

    def __getstate__(self):
        """Return state values to be pickled."""
        return None

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        pass


class MRIMemoryDataset(Dataset):
    def __init__(
        self,
        data_file_path: Path,
        header_file_path: Path,
        subject_list: np.ndarray,
        exclude: list[int] = [],
    ):
        """Create a dataset from the selected subjects in the subject list

        Args:
            data_file_path (Path): Data h5 file path.
            header_file_path (Path): Header csv file path.
            subject_list (np.ndarray): ist of all the subjects to include.
            exclude (list[int], optional): list of features to exclude from
            training. Defaults to [].
        """

        header = pd.read_csv(header_file_path, index_col=0).to_numpy()
        subject_list = np.array(subject_list)
        ind = header[np.isin(header[:,1],subject_list),0]
        
        #selection = np.arange(len(ind))
        selection = ind
        with h5py.File(data_file_path, "r") as archive:
            self.data = archive.get("data1")[selection]

        # delete excluded features
        self.data = np.delete(self.data, exclude, axis=1)

        self.data = torch.from_numpy(self.data).to("cuda")

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.data[index]

    def __getstate__(self):
        """Return state values to be pickled."""
        return None

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        pass


class MRIDataset(Dataset):
    def __init__(
        self,
        data_file_path: Path,
        header_file_path: Path,
        subject_list: np.ndarray,
        exclude: list[int] = [],
    ):
        """Create a dataset from the selected subjects in the subject list

        Args:
            data_file_path (Path): Data h5 file path.
            header_file_path (Path): Header csv file path.
            subject_list (np.ndarray): ist of all the subjects to include.
            exclude (list[int], optional): list of features to exclude from
            training. Defaults to [].
        """
        logger.warning(
            "MRIDataset is very slow compared to MRIMemoryDataset, only use MRIDataset if you don't have enough memory. "
            + "You can enable the use of MRIMemoryDataset by setting --in_memory in the console"
        )

        self.data_file_path = data_file_path
        self.header_file_path = header_file_path
        self.subject_list = subject_list
        self.exclude = exclude

        header = pd.read_csv(self.header_file_path, index_col=0).to_numpy()
        self.ind = header[np.isin(header[:,1],self.subject_list),0]
        
        self.selection = self.ind

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.selection)

    def __getitem__(self, index):
        """Generates one sample of data"""
        with h5py.File(self.data_file_path, "r") as archive:
            data = archive.get("data1")[self.selection[index]]
        data = np.delete(data, self.exclude)

        return data

    def __getstate__(self):
        """Return state values to be pickled."""
        return None

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        pass


class MRIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        subject_train,
        subject_val,
        data_file: Path,
        header_file: Path,
        batch_size: int = 256,
        in_memory: bool = False,
    ):
        """Collection of train and validation data sets.

        Args:
            data_dir (Path): Path to the data directory.
            data_file_name (str): file name of the H5 file.
            header_file_name (str): file name of the CSV file.
            batch_size (int, optional): training batch size. Defaults to 256.
            subject_train (list[int], optional): subjects to include in
            training. Defaults to [11, 12, 13, 14].
            subject_val (list[int], optional): subject(s) to include in
            validation. Defaults to [15].
            in_memory (bool): Whether to load the entire dataset in memory.
            Defaults to False.
        """
        super(MRIDataModule, self).__init__()

        self.data_file = data_file
        self.header_file = header_file
        self.batch_size = batch_size
        self.in_memory = in_memory
        
        # GENERAL DATA
        self.subject_train = np.array(subject_train)
        self.subject_val = np.array(subject_val)

        # Only assign 2 workers if Python is running on Windows (nt).
        #self.num_workers = 2 if os.name == "nt" else os.cpu_count()
        #self.num_workers = TO ADD

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Add model specific arguments to argparse.

        Args:
            parent_parser (ArgumentParser): parent argparse to add the new arguments to.

        Returns:
            ArgumentParser: parent argparse.
        """
        parser = parent_parser.add_argument_group("autoencoder.MRIDataModule")
        parser.add_argument(
            "--data_file",
            "-i",
            type=file_path,
            required=True,
            metavar="PATH",
            help="file name of the H5 file",
        )
        parser.add_argument(
            "--header_file",
            type=file_path,
            required=True,
            metavar="PATH",
            help="file name of the header file",
        )
        parser.add_argument(
            "--batch_size",
            default=256,
            type=int,
            metavar="N",
            help="input batch size for training (default: 64)",
        )
        parser.add_argument(
            "--val_subj",
            default=15,
            type=int,
            metavar="N",
            help="subject employed for validation (default: 15)",
        )
        parser.add_argument(
            "--in_memory",
            action="store_true",
            help="load the entire dataset into memory",
        )

        return parent_parser

    def setup(self, stage: Optional[str]) -> None:
        DatasetClass = MRIMemoryDataset if self.in_memory else MRIDataset

        self.train_set = DatasetClass(
            self.data_file,
            self.header_file,
            #self.batch_size,
            self.subject_train,
        )
        self.val_set = DatasetClass(
            self.data_file,
            self.header_file,
            #self.batch_size,
            self.subject_val,
        )

    def train_dataloader(self) -> DataLoader:
        if self.in_memory:
            return DataLoader(
                self.train_set, batch_size=self.batch_size, shuffle=True, 
            )
        else:
            return DataLoader(
                self.train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=True,
            )

    def val_dataloader(self) -> DataLoader:
        if self.in_memory:
            return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, drop_last=True,)
        else:
            return DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
            )

    def test_dataloader(self) -> DataLoader:
        if self.in_memory:
            return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, drop_last=True,)
        else:
            return DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True,
            )
