# CL+eq: physics-informed estimation and prediction

The objective of this task is to predict the MRI signal and estimate quantitative parameters from a biophysical model from a reduced specified MRI sub-protocol and reconstruct the whole dataset. The main files used for this task are [concrete_autoencoder_zebraorig.py](./src/autoencoder2/concrete_autoencoder_zebraorig.py), [trainer_onlyzebra.py](./trainer_files/onlyzebra/trainer_onlyzebra.py) for training the network, and [trainer_predict_zebra.py](./trainer_files/onlyzebra/trainer_predict_zebra.py) to estimate the quantitative maps and predict the MRI signal.

The trainer files are basically python scripts with the Main function to run the main code and organise the outputs. The other python file contains the following classes: `qmrizebra` and `ConcreteAutoencoder`.

* `qmrizebra` is the network employed to estimate the quantitative parameters and predict the MRI signal following the ZEBRA-based model. Its methods are:
  * `__init__()`: the constructor, where different parameters related to the neural network, the acquisition parameters and the MRI signal equation are defined.
  * `getparams()`: to get the quantitative MRI parameters through the application of the neural network.
  * `getnorm()`: to normalise the output from log activations of the neural network.
  * `getsignals()`: to get the signal values applying the equations from the ZEBRA-based model.
  * `forward()`: to run the network obtaining the quantitative parameters and the predicted MRI signal.

* `ConcreteAutoencoder` is not a concrete autoencoder. It is the class employed to initialise an object of the `qmrizebra` class to predict the MRI signal and estimate quantitative parameters according to the biophysical model with an MRI sub-protocol. Its methods are:
  * `__init__()`: the constructor, where the two networks are defined with some hyperparameters.
  * `add_model_specific_args()`: static method to introduce arguments when running the pertinent scripts (see [tutorial_cleq_onlyest.md](../tutorials/tutorial_cleq_onlyest.md) for details).
  * `configure_optimizers()`: to define the Adam optimiser.
  * `training_step()`: to get the loss value during training.
  * `validation_step()`: to get the loss value during validation.
  * `_shared_eval()`: to calculate the loss value on a batch.
  * `forward()`: to run the concrete layer and estimate the quantitative parameters and predict the MRI signal using the sub-selected measurements.
