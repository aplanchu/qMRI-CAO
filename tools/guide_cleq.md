# CL+eq: physics-informed selection sub-protocol

The main objective of this task is to extract the optimal MRI sub-protocol according to a specific number of volumes using a physics-informed network. The main files used for this task are [concrete_autoencoder_zebra_all_series2.py](./src/autoencoder2/concrete_autoencoder_zebra_all_series2.py), [trainer_v2.py](./trainer_files/zebra_ca/trainer_v2.py) for training the network, and [trainer_predict_v2.py](./trainer_files/zebra_ca/trainer_predict_v2.py) to estimate the quantitative maps and predict the MRI signal.

The trainer files are basically python scripts with the Main function to run the main code and organise the outputs. The other python file contains the following classes: `Encoder`, `qmrizebra` and `ConcreteAutoencoder`.

* `Encoder` is the feature selection encoder composed by the concrete layer. Its methods are:
  * `__init__()`: the constructor, where the temperature and regularisation parameters are defined with the initialisation of the concrete layer.
  * `update_temp()`: to update the temperature of the concrete autoencoder at each epoch.
  * `calc_mean_max()`: to calculate the mean of the maximum values of the softmax function of the logits that determine the selected measurements (used as stopping criterion)
  * `get_indexes()`: to get the selected measurements of the sub-protocol.
  * `regularization()`: to apply the regularisation that allows to extract a unique set of measurements of the specified size.
  * `forward()`: to run the concrete layer encoder.

* `qmrizebra` is the network employed to estimate the quantitative parameters and predict the MRI signal following the ZEBRA-based model. Its methods are:
  * `__init__()`: the constructor, where different parameters related to the neural network, the acquisition parameters and the MRI signal equation are defined.
  * `getparams()`: to get the quantitative MRI parameters through the application of the neural network.
  * `getnorm()`: to normalise the output from log activations of the neural network.
  * `getsignals()`: to get the signal values applying the equations from the ZEBRA-based model.
  * `forward()`: to run the network obtaining the quantitative parameters and the predicted MRI signal.

* `ConcreteAutoencoder` is the class employed to initialise objects of the `Encoder` and `qmrizebra` classes and apply the "two-network" scheme to select the optimal MRI-subprotocol according to the biophysical model. Its methods are:
  * `__init__()`: the constructor, where the two networks are defined with some hyperparameters.
  * `add_model_specific_args()`: static method to introduce arguments when running the pertinent scripts (see [tutorial_cleq.md](../tutorials/tutorial_cleq.md) for details).
  * `configure_optimizers()`: to define the Adam optimiser.
  * `training_step()`: to get the loss value during training.
  * `validation_step()`: to get the loss value during validation.
  * `on_train_epoch_start()`: to update the temperature of the concrete layer at the beginning of each epoch.
  * `on_epoch_end()`: to get the value used to define the stopping criterion.
  * `_shared_eval()`: to calculate the loss value on a batch.
  * `get_indices()`: to get the selected sub-measurements.
  * `forward()`: to run the concrete layer and estimate the quantitative parameters and predict the MRI signal using the sub-selected measurements.
