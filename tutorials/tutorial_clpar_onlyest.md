# Tutorial CL+par: estimation and prediction optimising the estimation of parameters

This tutorial shows how to train a network to estimate the quantitative parameters and predict the MRI signal using a ZEBRA-based biophysical representation and a predefined MRI sub-protocol, optimising the estimation of parameters instead of the prediction of the MRI signal.

This tutorial makes reference to the files [*script_clpar_onlyzebra*](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/template_scripts/script_clpar_onlyzebra) and [*script_clpar_onlyzebra_predict*](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/template_scripts/script_clpar_onlyzebra_predict). For each script, the relevant arguments are briefly described and the example available in each script is shown to better understand the running of the script. Diverse categories for the arguments are also shown:

1. **INPUT**. Input file.

2. **OUTPUT**. Output file or folder.

3. **EDIT**. Editable value to adjust training parameters.

## Training for estimation of quantitative parameters and prediction of the MRI signal

This part makes reference to [*script_clpar_onlyzebra*](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/template_scripts/script_clpar_onlyzebra). The arguments of the script are:

* `data_file` (**INPUT**): h5 file that contains the dataset (MRI signal) and the ground truth parameters for all the subjects.

* `header_file` (**INPUT**): csv file that contains the header.

* `latent_size` (**EDIT**, optional): integer showing the number of selected measurements or the size of the sub-selected protocol (default = 500).

* `latent_size2` (**EDIT**, optional): integer showing the number of parameters of the employed model (default = 7 for the ZEBRA-based model).

* `encoder2_hidden_layers` (**EDIT**, optional): integer showing the number of hidden layers in the neural network estimating the parameters from the sub-selected protocol. A value of 2 (default) was used to test the approach.

* `ind_path` (**INPUT**): string indicating the path of the txt file where the values of the rows reflecting the selected sub-protocol are saved.

* `gt_path` (**EDIT**): string indicating the path of the nifti file of the parameters used as ground truth for the validation subject.

* `learning_rate` (**EDIT**, optional): value showing the learning rate (default = 1e-3).

* `gpus` (**EDIT**, optional): integer showing the number of GPUs employed to train the network (default = 1). The method may not work for 2 or more GPUs.

* `max_epochs` (**EDIT**): integer showing the maximum number of epochs employed to train the network. For the MUDI data, 100 was the employed value to test the approach, although it is recommended to use a higher value for a relative low number of selected measurements.

* `batch_size` (**EDIT**, optional): integer showing the number of voxels per MRI volume included in the batch. A value of 256 (default) was used to test the approach.

* `val_subj` (**EDIT**, optional): integer showing the ID of the subject used for validation following a leave-one-out approach (default = 15).

* `seed_number` (**EDIT**, optional): integer showing a seed initialisation (default = 42).

* `folder_hyperparams` (**OUTPUT**): string indicating the path where the folder with the hyperparameters and diverse training results, including the hyperparameters of the best model, are saved. Values from this folder are used in [*script_clpar_onlyzebra_predict*](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/template_scripts/script_clpar_onlyzebra_predict).

* `mridata_path` (**INPUT**): string indicating the path of the txt file where the values of the acquisition parameters are shown.

* `prct_norm` (**EDIT**, optional): value that shows the quantile used as maximum value to normalise the quantitative parameters (default = 0.975). This value should be between 0 and 1.

* `stopping_threshold` (**EDIT**, optional): value indicating the minimum validation loss value (mean-squared error) as stopping criterion alternatively to the maximum number of epochs (default = 1e-5).

* `in_memory` (**EDIT**): boolean indicating whether GPU (true) is used to train the method.

Assuming that we are in the folder where the ["tools"](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/) directory is saved and that we are using the environment available in the root directory of the project, an example is shown below for the MUDI dataset, 500 subselected measurements and the ZEBRA-based model:

```
$ python trainer_files/onlyzebra_evmaps/trainer_onlyzebra.py \
    --data_file PATH/data_.hdf5 \
    --header_file PATH/header_.csv \
    --latent_size 500 \
    --latent_size2 7 \
    --encoder2_hidden_layers 2 \
    --ind_path PATH/INDICES.txt \
    --gt_path PATH/gt_params_subj11.nii.gz \
    --learning_rate 1e-3 \
    --gpus=1 \
    --max_epochs 100 \
    --batch_size 256 \
    --val_subj 11 \
    --seed_number 42 \
    --folder_hyperparameters PATH_hparams/hyperparams_clpar_onlyzebra \
    --mridata_path PATH/parameters_new.txt \
    --prct_norm 0.975 \
    --stopping_threshold 1e-5 \ 
    --in_memory
```

## Saving the estimated parameters and predicted MRI signal

This part makes reference to [*script_clpar_onlyzebra_predict*](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/template_scripts/script_clpar_onlyzebra_predict). The additional arguments of the script are:

* `input_output_size` (**EDIT**, optional): integer showing the total number of MRI measurements (default = 1344 for the MUDI dataset).

* `mask_file` (**INPUT**, optional): nifti file with the mask of the validation subject.

* `path_save_param` (**OUTPUT**): txt or nifti (if a mask is provided) file where the estimated paratemers for the validation subject are saved. Another file with the same name and *_dperp* is created for the perpendicular diffusivity.

* `path_save` (**OUTPUT**): txt or nifti (if a mask is provided) file where the predicted MRI signal for all volumes of the validation subject is saved.

* `hparams` (**EDIT**): *hparams.yaml* file with the saved hyperparameters from the first training step ([*script_clpar_onlyzebra*](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/template_scripts/script_clpar_onlyzebra)).

* `checkpoint` (**EDIT**): ckpt file from the saved hyperparameters of the first training step.

The two last parameters can be used as well in the first training step.

Assuming that we are in the folder where the ["tools"](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/) directory is saved and that we are using the environment available in the root directory of the project, an example is shown below for the MUDI dataset, 500 subselected measurements and the ZEBRA-based model:

```
$ python trainer_files/onlyzebra_evmaps/trainer_predict_zebra.py \
    --data_file PATH/data_.hdf5 \
    --header_file PATH/header_.csv \
    --input_output_size 1344 \
    --latent_size 500 \
    --latent_size2 7 \
    --encoder2_hidden_layers 2 \
    --ind_path PATH/INDICES.txt \
    --gt_path PATH/gt_params_subj11.nii.gz \
    --learning_rate 1e-3 \
    --gpus=1 \
    --max_epochs 1 \
    --batch_size 256 \
    --val_subj 11 \
    --seed_number 42 \
    --mask_file PATH/MASK_NAME.nii.gz \
    --path_save_param PATH/PARAM_SIG_NAME.nii.gz \
    --path_save PATH/PRED_SIG_NAME.nii.gz \
    --folder_hyperparameters PATH_hparams/hyperparams_clpar_onlyzebra_pred \
    --mridata_path PATH/parameters_new.txt \
    --prct_norm 0.975
    --hparams PATH_hparams/hyperparams_clpar_onlyzebra/hparams.yaml \
    --checkpoints PATH_hparams/hyperparams_clpar_onlyzebra/checkpoints/epoch=XXXX-step=XXXXXX.ckpt \
    --in_memory
```

NOTE: if prediction from a test subject not included in the training or validation datasets is desired, the *data_file* and *header_file* parameters should be changed to a dataset that additionally incorporates the test subject, and the parameter *val_subj* should be changed to the ID of the test subject.
