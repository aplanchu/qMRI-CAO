from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.profiler import PassThroughProfiler
from torch import nn
import pandas as pd

from torch import reshape as tshape
from torch import cat as tcat
from torch import exp as texp
from torch import log as tlog
from torch import abs as tabs
from torch import erf as terf
from torch import sqrt as tsqrt
from torch import matmul as tmat

import sys
from pathlib import Path
sys.path.insert(0,str(Path().absolute()))

from argparse2 import file_path
from logger import logger


class qmrizebra(pl.LightningModule):
    def __init__(
        self,
        #pdrop, #mridata, 
        #input_size1: int,
        input_size2: int,
        output_size2: int,
        n_hidden_layers2: int,
        mridata_path: str,
    ):
        """Network to predict quantitative parameters using the ZEBRA-based model (Hutter et al., 2018; Tax et al., 2021)

        Args:
            input_size2 (int): size of the input layer. Should be the same as the `output_size` of the Encoder class (number of selected measurements by the concrete autoencoder)
            output_size2 (int): size of the latent layer. Should be the number of quantitative parameters to be estimated (7 for the ZEBRA-based model)
            n_hidden_layers2: number of hidden layers
            mridata_path: path to the file (txt) with the acquisition parameters
        """

        super(qmrizebra, self).__init__()
        
        self.mridata = pd.read_csv(mridata_path, sep=" ", header=None)
        # DESIGNED FOR MUDI DATA, CHANGE IF NECESSARY AND ACCORDING TO THE ACQUISITION PARAMETERS
        self.mridata.columns = ["x", "y", "z", "bval","TI","TD"]
        self.mridata.TD = self.mridata.TD - min(self.mridata.TD) # TD = TE - min(TE)
        self.mridata["TR"] = 7500.0
        
        self.npars = output_size2 # Number of parameters (7 in the ZEBRA-based model)
        
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.mriseq = torch.transpose(torch.tensor(self.mridata.values.astype(np.float32)),0,1)
        
        indices2 = np.arange(2 + n_hidden_layers2)
        data_indices2 = np.array([indices2[0], indices2[-1]])
        data2 = np.array([input_size2, output_size2])

        layer_sizes = np.interp(indices2, data_indices2, data2).astype(int)
        n_layers = len(layer_sizes)#+1
        
        # Construct the network
        layers = OrderedDict()
        for i in range(1, n_layers):
            n = i - 1
            layers[f"linear_{n}"] = nn.Linear(layer_sizes[i - 1], layer_sizes[i])
            layers[f"relu_{n}"] = nn.ReLU(True)
            if i == n_layers - 1:  # Last layer
                layers[f"softplus_{n}"] = nn.Softplus()
                

        logger.debug("encoder2 layers: %s", layers)

        self.encoder2 = nn.Sequential(layers)
        
        # Add learnable normalisation factors to convert output neuron activations to tissue parameters
        normlist = []
        for pp in range(output_size2):
            normlist.append(nn.Linear(1, 1, bias = False))
        self.sgmnorm = nn.ModuleList(normlist)
        
        ### Set the possible maximum and minimum value of the output parameters
        theta_min = 0.0 # in rad
        phi_min = 0.0 # in rad
        dpar_min = 0.01 # um^2/ms from Grussu et al. (qmrinet)
        kperp_min = 0 # To make sure that dperp <= dpar. dperp = kperp*dpar
        t2star_min = 0.01 # in ms (https://mri-q.com/why-is-t1--t2.html), a bit lower than T2 bound for proteins
        t1_min = 100 # in ms at 3T from Grussu et al. (qmrinet)
        s0_min = 0.5 # from Grussu et al. (qmrinet)
        
        theta_max = np.pi # in rad
        phi_max = 2*np.pi-0.01 # in rad, a bit lower than 2pi
        dpar_max = 3.2 # um^2/ms from Grussu et al. (qmrinet)
        kperp_max = 1 # To make sure that dperp <= dpar. dperp = kperp*dpar
        t2star_max = 2000 # in ms, CSF value (https://mri-q.com/why-is-t1--t2.html)
        t1_max = 5000 # in ms, from the ZEBRA paper
        s0_max = 5 # from Grussu et al. (qmrinet)
        
        self.param_min = torch.tensor([theta_min, phi_min, dpar_min, kperp_min, t2star_min, t1_min, s0_min])
        self.param_max = torch.tensor([theta_max, phi_max, dpar_max, kperp_max, t2star_max, t1_max, s0_max])
        self.param_name = ['theta', 'phi', 'dpar', 'kperp', 't2star', 't1', 's0']
        
        self.con_one = torch.tensor([1.0])
        self.con_two = torch.tensor([2.0])
        self.Nmeas = self.mriseq.shape[1]
        self.b_delta = torch.tensor([1.0])
        self.ones_ten = torch.ones(1,self.Nmeas)
    
    def getnorm(self, x):
        """ Get the output from log activations of the encoder and normalise it
        
            u_out = mynet.getnorm(u_in)
            
            * mynet: initialised qmrizebra
            
            * u_in: Tensor storing the output neuronal activations
                    Nvoxels x Nparams_to_estimate for a mini-batch
                    
            * u:out = normalised u_in
        """
        
        if x.dim()==1:
            
            # 1D tensor corresponding to a voxel
            normt = torch.zeros(self.npars)
            for pp in range(self.npars):
                bt = torch.zeros(self.npars)
                bt[pp] = 1.0
                con_one = torch.tensor([1.0])
                bt = self.sgmnorm[pp](con_one)*bt
                normt = normt + bt
                
            # Normalise
            normt = tabs(normt)
            x = x*normt
            
        elif x.dim()==2:
            
            # Tensor with Nvoxels x Nparams
            normt = torch.zeros(x.shape[0],self.npars)
            for pp in range(self.npars):
                bt = torch.zeros(x.shape[0],self.npars)
                bt[:,pp] = 1.0
                bt = torch.tensor(self.sgmnorm[pp](self.con_one))*bt
                normt = normt + bt
        
            # Normalise
            normt = tabs(normt)
            x = x*normt
            
        else:
            raise RuntimeError('getnorm() only accepts 1D or 2D inputs')
            
        return x
    
    def getparams(self, x):
        """ Get tissue parameters from initialised qmrizebra
        
            p = mynet.getparams(x_in)
            
            * mynet: initialised qmrizebra
            
            * xin: Tensor storing MRI measurements (a voxel or a mini-batch)
                    voxels x Nmeasurements in a mini-batch
                    
            * p:   Tensor storing the predicted parameters
                   Nvoxels x Nparams_to_estimate for a mini-batch            
        """
        
        x = self.encoder2(x) # the last layer cannot be 0.0 due to the softplus
        
        ## Normalise neuronal activations in [0; 1.0] in 4 steps (A, B, C, D):
        x = tlog(x) # 1. Log activations
        #con_two = torch.tensor([2.0])
        #con_two = con_two#.type_as(x)
        x = x - tlog(tlog(self.con_two)) # 2. We remove possible negative values considering that the minumum of x is log(2.0)
        x = self.getnorm(x) # 3. Normalisation using a multiplying learnable factor
        x = 2.0*(1.0 / (1.0 + texp(-x)) - 0.5) # 4. Sigmoid function
        ## Map normalised neuronal activations to MRI tissue parameter ranges \
        
        # Single voxels
        if x.dim()==1:
            for pp in range(0,self.npars):
                x[pp] = (self.param_max[pp] - self.param_min[pp])*x[pp] + self.param_min[pp]
                
        # Mini-batch
        elif x.dim()==2:
            t_ones = torch.ones(x.shape[0],1)
            
            max_val = torch.cat( ( self.param_max[0]*t_ones , self.param_max[1]*t_ones , self.param_max[2]*t_ones , self.param_max[3]*t_ones, self.param_max[4]*t_ones, self.param_max[5]*t_ones , self.param_max[6]*t_ones), 1  ) 
                             
            min_val = torch.cat( ( self.param_min[0]*t_ones , self.param_min[1]*t_ones , self.param_min[2]*t_ones , self.param_min[3]*t_ones, self.param_min[4]*t_ones, self.param_min[5]*t_ones , self.param_min[6]*t_ones), 1   ) 
            
            x = (max_val - min_val)*x + min_val
                             
        return x
    
    # Decoder: Estimate the signal from MRI parameters estimated with the encoder
    def getsignals(self, x):
        """ Use the ZEBRA model to obtain the diffusion signal.
        
            x_out = mynet.getsignals(p_in)
            
            * mynet: initialised qmrizebra
            
            * p_in:   Tensor storing the predicted parameters
                   Nvoxels x Nparams_to_estimate for a mini-batch 
            
            * x_out: Tensor storing the predicted MRI signals according to ZEBRA
                    Nvoxels x Nparams_to_estimate for a mini-batch 
        """
                             
        ## Compute MRI signals from input parameter x (microstructural tissue parameters)
        x1 = self.mriseq[0,:]
        x2 = self.mriseq[1,:]
        x3 = self.mriseq[2,:]
        bval = self.mriseq[3,:]
        bval = bval / 1000.0 # from s/mm^2 to ms/um^2
        TI = self.mriseq[4,:]
        TD = self.mriseq[5,:]
        TR = self.mriseq[6,:]
        # we assume that b_delta = 1
        
        if x.dim()==1: # Updated equation in the second term of b_D (Dpar+2Dperp instead of Dperp+2Dpar to compensate underestimated diffusivities from the equation displayed in the reference)
            b_D = self.b_delta / 3.0 * bval * (x[2] - x[3]*x[2]) - bval / 3.0 * (x[2] + 2.0 * x[3]*x[2]) - bval * self.b_delta * (torch.square(torch.dot([x1,x2,x3],[torch.cos(x[1])*torch.sin(x[0]),torch.sin(x[0])*torch.sin(x[1]),torch.cos(x[0])])) * (x[2] - x[3]*x[2]))
            s_tot = x[6] * texp(b_D) * tabs(1.0 - 2.0 * texp(-TI/x[5]) + texp(-TR/x[5])) * texp(-TD/x[4])
            x = 1.0*s_tot
            return x
            
        elif x.dim()==2:
            Nvox = x.shape[0]
            x1 = tshape(x1, (1,self.Nmeas))
            x2 = tshape(x2, (1,self.Nmeas))
            x3 = tshape(x3, (1,self.Nmeas))
            bval = tshape(bval, (1,self.Nmeas))
            TI = tshape(TI, (1,self.Nmeas))
            TD = tshape(TD, (1,self.Nmeas))
            TR = tshape(TR, (1,self.Nmeas))
            
            b_D = self.b_delta / 3.0 * tmat(tshape(x[:,2] - x[:,3]*x[:,2], (Nvox,1)), bval)
            b_D = b_D - 1.0 / 3.0 * tmat(tshape(x[:,2] + 2.0 * x[:,3]*x[:,2], (Nvox,1)), bval) # Updated (Dpar+2Dperp instead of Dperp+2Dpar to compensate underestimated diffusivities from the equation displayed in the reference)
            angles_dprod = tmat(tshape(torch.sin(x[:,0])*torch.cos(x[:,1]), (Nvox,1)), x1) + tmat(tshape(torch.sin(x[:,1])*torch.sin(x[:,0]), (Nvox,1)), x2) + tmat(tshape(torch.cos(x[:,0]), (Nvox,1)), x3)
            b_D_term3 = tmat(tshape(x[:,2] - x[:,3]*x[:,2], (Nvox,1)), bval) * torch.square(angles_dprod)
            b_D = b_D - self.b_delta * b_D_term3

            s_tot = tmat(tshape(x[:,6],(Nvox,1)), self.ones_ten)
            
            s_tot = s_tot * texp(b_D)
            s_tot = s_tot * tabs(1.0 - 2.0 * texp(tmat(tshape(1.0/x[:,5],(Nvox,1)),-TI)) + texp(tmat(tshape(1.0/x[:,5],(Nvox,1)),-TR)))
            s_tot = s_tot * texp(tmat(tshape(1.0/x[:,4],(Nvox,1)),-TD))
            
            x = 1.0*s_tot
            return x
    
    def forward(self, x: torch.Tensor):# -> torch.Tensor:
        """Uses the trained decoder to make inferences.

        Args:
            x (torch.Tensor): input data. Should be the same size as the decoder input.

        Returns:
            torch.Tensor: decoder output of size `output_size`.
        """
        encoded2 = self.getparams(x)
        decoded = self.getsignals(encoded2)
        return encoded2, decoded

class ConcreteAutoencoder(pl.LightningModule):
    def __init__(
        self,
        ind_path: Path,
        mridata_path: str,
        latent_size: int = 500,
        latent_size2: int = 7,
        encoder2_hidden_layers: int = 2,
        learning_rate: float = 1e-3,
        profiler=None,
    ) -> None:
        """Trains a network to estimate the parameters of a specified model (ZEBRA as default) and predict the signal following the model

        Args:
            ind_path: path to the file (txt) with the selected measurements
            mridata_path: path to the file (txt) with the acquisition parameters
            latent_size (int): number of selected measurements and input size of the neural network to estimate the quantitative maps from the model
            latent_size2 (int): number of estimated parameters and output size of the neural network to estimate the quantitative maps from the model
            encoder2_hidden_layers (int, optional): number of hidden layers for the encoder to estimate the parameters of the model. Defaults to 2.
            learning_rate (float, optional): learning rate for the optimizer. Defaults to 1e-3.
        """
        super(ConcreteAutoencoder, self).__init__()
        self.save_hyperparameters()

        self.qmrizebra = qmrizebra(
            #input_size1 = input_output_size,
            input_size2 = latent_size,
            output_size2 = latent_size2,
            n_hidden_layers2 = encoder2_hidden_layers,
            mridata_path = mridata_path,
        )
        
        self.learning_rate = learning_rate
        self.ind_mudi = np.loadtxt(ind_path)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Add model specific arguments to argparse.

        Args:
            parent_parser (ArgumentParser): parent argparse to add the new arguments to.

        Returns:
            ArgumentParser: parent argparse.
        """
        parser = parent_parser.add_argument_group("autoencoder.ConcreteAutoencoder")
        parser.add_argument(
            "--checkpoint",
            default=None,
            type=file_path,
            metavar="PATH",
            help="Checkpoint file path to restore from.",
        )
        parser.add_argument(
            "--hparams",
            default=None,
            type=file_path,
            metavar="PATH",
            help="hyper parameter file path to restore from.",
        )
        parser.add_argument(
            "--input_output_size",
            "-s",
            default=1344,
            type=int,
            metavar="N",
            help="size of the input and output layer",
        )
        parser.add_argument(
            "--latent_size",
            "-l",
            default=500,
            type=int,
            metavar="N",
            help="size of latent layer",
        )
        parser.add_argument(
            "--latent_size2",
            "-l2",
            default=7,
            type=int,
            metavar="N",
            help="size of latent layer 2",
        )
        parser.add_argument(
            "--encoder2_hidden_layers",
            default=2,
            type=int,
            metavar="N",
            help="number of hidden layers for the second encoder (default: 2)",
        )
        parser.add_argument(
            "--learning_rate",
            type=float,
            default=1e-3,
            metavar="N",
            help="learning rate for the optimizer (default: 1e-2)",
        )

        parser.add_argument(
            "--ind_path",
            default=None,
            type=file_path,
            metavar="PATH",
            help="File with the selected measurements",
        )
        
        parser.add_argument(
            "--mridata_path",
            default=None,
            type=str,
            metavar="PATH",
            help="path with the acquisition parameters (txt file)",
        )

        return parent_parser

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Uses the trained autoencoder to make inferences.

        Args:
            x (torch.Tensor): input data. Should be the same size as encoder input.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (encoder output, decoder output)
        """
        decoded, decoded2 = self.qmrizebra(x[:,self.ind_mudi]) # Actually, this decoded should be "encoded", but it has the name "decoded" to be returned together with the predicted signal and respect the original notation
        return decoded, decoded2

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self._shared_eval(batch, batch_idx, "train")

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_eval(batch, batch_idx, "val")

    def _shared_eval(
        self, batch: torch.Tensor, batch_idx: int, prefix: str
    ) -> torch.Tensor:
        """Calculate the loss for a batch.

        Args:
            batch (torch.Tensor): batch data.
            batch_idx (int): batch id.
            prefix (str): prefix for logging.

        Returns:
            torch.Tensor: calculated loss.
        """
        decoded, decoded2 = self.forward(batch)
        loss = F.mse_loss(decoded2, batch)

        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
