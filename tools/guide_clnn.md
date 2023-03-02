# CL+NN: full data-driven selection sub-protocol

The main objective of this task is to extract the optimal MRI sub-protocol according to a specific number of volumes using a full data-driven approach. The main files used for this task are [concrete_autoencoder_orig2.py](./src/autoencoder2/concrete_autoencoder_orig2.py), [trainer.py](./trainer_files/only_ca/trainer.py) for training the network, and [trainer_predict.py](./trainer_files/only_ca/trainer_predict.py) to predict the MRI signal.

The trainer files are basically python scripts with the Main function to run the main code and organise the outputs. The other python file contains the following classes: `Encoder`, `Decoder` and `ConcreteAutoencoder`.

* `Encoder` is the feature selection encoder composed by the concrete layer. Its methods are:
  * `__init__()`: the constructor, where the temperature and regularisation parameters are defined with the initialisation of the concrete layer.
  * `update_temp()`: to update the temperature of the concrete autoencoder at each epoch.
  * `calc_mean_max()`: to calculate the mean of the maximum values of the softmax function of the logits that determine the selected measurements (used as stopping criterion)
  * `get_indexes()`: to get the selected measurements of the sub-protocol.
  * `regularization()`: to apply the regularisation that allows to extract a unique set of measurements of the specified size.
  * `forward()`: to run the concrete layer encoder.

* `Decoder` is a standard decoder to upsample the MRI volumes from the sub-selected protocol to the whole dataset. Its methods are:
  * `__init__()`: the constructor, where the parameters of the neural network with its hidden layers are defined.
  * `forward()`: to run the decoder.

* `ConcreteAutoencoder` is the class employed to initialise objects of the `Encoder` and `Decoder` classes to select the optimal MRI-subprotocol. Its methods are:
  * `__init__()`: the constructor, where the encoder and decoder are defined with some hyperparameters.
  * `add_model_specific_args()`: static method to introduce arguments when running the pertinent scripts (see [tutorial_clnn.md](../tutorials/tutorial_clnn.md) for details).
  * `configure_optimizers()`: to define the Adam optimiser.
  * `training_step()`: to get the loss value during training.
  * `validation_step()`: to get the loss value during validation.
  * `on_train_epoch_start()`: to update the temperature of the concrete layer at the beginning of each epoch.
  * `on_epoch_end()`: to get the value used to define the stopping criterion.
  * `_shared_eval()`: to calculate the loss value on a batch.
  * `get_indices()`: to get the selected sub-measurements.
  * `forward()`: to run the concrete layer and estimate the quantitative parameters and predict the MRI signal using the sub-selected measurements.

