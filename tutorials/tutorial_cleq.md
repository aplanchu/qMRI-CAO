# Tutorial CL+eq: physics-informed selection sub-protocol

This tutorial shows how to train a network to select the optimal subset of MRI measurements with a physics-informed approach combining concrete autoencoders and a ZEBRA-based biophysical representation.

This tutorial makes reference to the files [*script_indices_cleq*](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/template_scripts/script_indices_cleq) and [*script_indices_cleq_predict*](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/template_scripts/script_indices_cleq_predict). For each script, the relevant arguments are briefly described and the example available in each script is shown to better understand the running of the script. Diverse categories for the arguments are also shown:

1. **INPUT**. Input file.

2. **OUTPUT**. Output file or folder.

3. **EDIT**. Editable value to adjust training parameters.

## Training and selection of the optimal sub-selection protocol

This part makes reference to [*script_indices_cleq*](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/template_scripts/script_indices_cleq). The arguments of the script are:

* `data_file` (**INPUT**): h5 file that contains the dataset.

* `header_file` (**INPUT**): csv file that contains the header.

* `input_output_size` (**EDIT**, optional): integer showing the total number of MRI measurements (default = 1344 for the MUDI dataset).

* `latent_size` (**EDIT**, optional): integer showing the number of selected measurements or the size of the sub-selected protocol (default = 500).

* `latent_size2` (**EDIT**, optional): integer showing the number of parameters of the employed model (default = 7 for the ZEBRA-based model).

* `encoder2_hidden_layers` (**EDIT**, optional): integer showing the number of hidden layers in the neural network estimating the parameters from the sub-selected protocol. A value of 2 (default) was used to test the approach.

* `learning_rate` (**EDIT**, optional): value showing the learning rate (default = 1e-3).

* `max_temp` (**EDIT**, optional): maximum and initial value of the temperature of the concrete layer (default = 10.0).

* `min_temp` (**EDIT**, optional): minimum value of the temperature of the concrete layer (default = 0.1).

* `gpus` (**EDIT**, optional): integer showing the number of GPUs employed to train the network (default = 1). The method may not work for 2 or more GPUs.

* `max_epochs` (**EDIT**): integer showing the maximum number of epochs employed to train the network. For the MUDI data, 5000 was the employed value to test the approach.

* `reg_lambda` (**EDIT**, optional): value showing the first regularisation parameter to obtain a unique number of different measurements according to the `latent_size` value. A value of 0.0 (default) reflects no regularisation. A value of 0.1 was used to test the approach.

* `reg_threshold` (**EDIT**, optional): value showing the second regularisation parameter to obtain a unique number of different measurements according to the `latent_size` value. A value of 0.0 reflects no regularisation. A value of 1.0 (default) was used to test the approach.

* `batch_size` (**EDIT**, optional): integer showing the number of voxels per MRI volume included in the batch. A value of 256 (default) was used to test the approach.

* `val_subj` (**EDIT**, optional): integer showing the ID of the subject used for validation following a leave-one-out approach (default = 15).

* `seed_number` (**EDIT**, optional): integer showing a seed initialisation (default = 42).

* `path_save_ind` (**OUTPUT**): string indicating the path of the txt file where the values of the rows reflecting the selected sub-protocol are saved after the training.

* `folder_hyperparams` (**OUTPUT**): string indicating the path where the folder with the hyperparameters and diverse training results, including the hyperparameters of the best model, are saved. Values from this folder are used in [*script_indices_cleq_predict*](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/template_scripts/script_indices_cleq_predict).

* `mridata_path` (**INPUT**): string indicating the path of the txt file where the values of the acquisition parameters are shown.

* `stopping_threshold` (**EDIT**, optional): value indicating the mean maximum value related to the concrete layer used as stopping criterion alternatively to the maximum number of epochs (default = 0.998).

* `in_memory` (**EDIT**): boolean indicating whether GPU (true) is used to train the method.

Assuming that we are in the folder where the ["tools"](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/) directory is saved and that we are using the environment available in the root directory of the project, an example is shown below for the MUDI dataset, 500 subselected measurements and the ZEBRA-based model:

```
$ python trainer_files/zebra_ca/trainer_v2.py \
    --data_file PATH/data_.hdf5 \
    --header_file PATH/header_.csv \
    --input_output_size 1344 \
    --latent_size 500 \
    --latent_size2 7 \
    --encoder2_hidden_layers 2 \
    --learning_rate 1e-3 \
    --max_temp 10.0 \
    --min_temp 0.1 \
    --gpus=1 \
    --max_epochs 5000 \
    --reg_lambda 0.1 \
    --reg_threshold 1.0 \
    --batch_size 256 \
    --val_subj 11 \
    --seed_number 42 \
    --path_save_ind PATH/ind_cleq.txt \
    --folder_hyperparameters PATH_hparams/hyperparams_cleq \
    --mridata_path PATH/parameters_new.txt \
    --stopping_threshold 0.998 \
    --in_memory
```

## Saving the estimated parameters and predicted MRI signal

This part makes reference to [*script_indices_cleq_predict*](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/template_scripts/script_indices_cleq_predict). In comparison with the previous script, the argument `path_save_ind` is no longer used. The additional arguments of the script are:

* `mask_file` (**INPUT**, optional): nifti file with the mask of the validation subject.

* `path_save_param` (**OUTPUT**): txt or nifti (if a mask is provided) file where the estimated paratemers for the validation subject are saved. Another file with the same name and *_dperp* is created for the perpendicular diffusivity.

* `path_save` (**OUTPUT**): txt or nifti (if a mask is provided) file where the predicted MRI signal for all volumes of the validation subject is saved.

* `hparams` (**EDIT**): *hparams.yaml* file with the saved hyperparameters from the first training step ([*script_indices_cleq*](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/template_scripts/script_indices_cleq)).

* `checkpoint` (**EDIT**): ckpt file from the saved hyperparameters of the first training step.

The two last parameters can be used as well in the first training step, although this is not recommended as the temperature parameters of the concrete autoencoder would be the same as for the first epoch (i.e., different to the last saved epoch) and hence the training of the network would not be a continuation of the previous running.

Assuming that we are in the folder where the ["tools"](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/) directory is saved and that we are using the environment available in the root directory of the project, an example is shown below for the MUDI dataset, 500 subselected measurements and the ZEBRA-based model:

```
$ python trainer_files/zebra_ca/trainer_predict_v2.py \
    --data_file PATH/data_.hdf5 \
    --header_file PATH/header_.csv \
    --input_output_size 1344 \
    --latent_size 500 \
    --latent_size2 7 \
    --encoder2_hidden_layers 2 \
    --learning_rate 1e-3 \
    --max_temp 10.0 \
    --min_temp 0.1 \
    --gpus=1 \
    --max_epochs 1 \
    --reg_lambda 0.1 \
    --reg_threshold 1.0 \
    --batch_size 256 \
    --val_subj 11 \
    --seed_number 42 \
    --mask_file PATH/MASK_NAME.nii.gz \
    --path_save_param PATH/PARAM_SIG_NAME.nii.gz \
    --path_save PATH/PRED_SIG_NAME.nii.gz \
    --folder_hyperparameters PATH_hparams/hyperparams_cleq_pred \
    --mridata_path PATH/parameters_new.txt \
    --hparams PATH_hparams/hyperparams_cleq/hparams.yaml \
    --checkpoint PATH_hparams/hyperparams_cleq/checkpoints/epoch=XXXX-step=XXXXXX.ckpt \
    --in_memory
```

NOTE: if prediction from a test subject not included in the training or validation datasets is desired, the *data_file* and *header_file* parameters should be changed to a dataset that additionally incorporates the test subject, and the parameter *val_subj* should be changed to the ID of the test subject.
