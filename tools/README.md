# User guide

This user guide briefly describes the python classes and functions used to carry out the diverse tasks. The tutorials showing how to run the code are in the "[tutorials](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/)" folder. All the code or scripts shown in this folder must be run in the path containing this folder. The different guides are:

* [**Guide CL+NN**](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/guide_clnn.md) describes the classes and functions used for the extraction of the sub-selected protocol and the prediction of the whole dataset with the upsampled MRI volumes with CL+NN, i.e., the full data-driven method.

* [**Guide CL+eq**](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/guide_cleq.md) describes the classes and functions used for the extraction of the sub-selected protocol with CL+eq, i.e., the physics-informed method.

* [**Guide CL+par**](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/guide_clpar.md) describes the classes and functions used for the extraction of the subselected protocol with CL+par, i.e., the method using the quantitative parameters in the loss function.

* [**Guide CL+eq estimation**](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/guide_cleq_onlyest.md) describes the classes and functions used for the estimation of the quantitative parameters and the prediction of the MRI signal from a specific sub-selected acquisition protocol optimising the signal prediction.

* [**Guide CL+par estimation**](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/guide_clpar_onlyest.md) describes the classes and functions used for the estimation of the quantitative parameters and the prediction of the MRI signal from a specific sub-selected acquisition protocol optimising the parameter estimation.

## Additional generated files

It is important to note some files that are generated when running diverse scripts.

### mlruns

For each run, a new folder is created in the directory [*mlruns/0/*](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/mlruns/0/) (it could also be [*1*](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/mlruns/0/) instead of *0*). The name of the generated folder is a sequence of characters (numbers and letters) without an obvious pattern to follow in relation to the running script. Thus, to associate the content of this folder to a particular script, this folder should be checked. In any case, this content is not important to obtain the main results and it can be checked in relation to some training parameters and metrics. Each generated folder contains the following subdirectories:

* **metrics**. This folder contains different files that show some metrics associated with the concrete autoencoder training, such as the temperature through epochs, or the training and validation loss through epochs.

* **params**. This folder contains different files with the values of diverse arguments employed for training each method, including those described in the "[tutorials](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/README.md)" and other hyperparameters or parameters related to the training (e.g., tpu cores, which have not been employed for the development of this project).

* **tags and artifacts**. These folders usually contain no relevant information or are empty.

### logs

There are other files generated when running the scripts in the *logs* directory. In the current version of the project, the created files have the name of *MUDI.log* or longer variants. These files just display the arguments used for each running script, similarly to the files mentioned previously in relation to the *params* folder. To modify the name of "MUDI" to anyone selected by the user, in the file [logger.py](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/src/autoencoder2/logger.py) just change the value of the parameter "LOGGER_NAME" to any preferred alternative.

Furthermore, for each running script without errors before starting the first epoch, a folder with name *version_N* is generated in the directory *logs/concrete_autoencoder*, being N an integer of value 0 or the next available integer not present in any folder name. The generated folder is equivalent to the *folder_hyperparameters* parameter described in the "[tutorials](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/README.md)", where more information is available. If there is no error when running the script, this folder is moved to the path specified in this parameter, and otherwise stays in the *logs/concrete_autoencoder* directory. It is recommended to remove this folder in case of failed runs to avoid any problem with future scripts.

### Other python files (src/autoencoder2/)

Some python files in the directory *./src/autoencoder2/* not shown in the user guide files are described below:

* [`argparse2.py`](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/src/autoencoder2/argparse2.py). This file contains functions to employ paths or specific files as arguments in the diverse scripts.

* [`dataset2.py`](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/src/autoencoder2/dataset2.py). This file allows to use a specific database and GPU, specifically an h5 file with a csv header. It is important to note that the number of employed CPUs for pytorch generators can be modified setting the value of the parameter *self.num_workers* in the class **MRIDataModule** and removing the corresponding commented code. 

* [`dataset2cpu.py`](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/src/autoencoder2/dataset2cpu.py). This file makes the same function of [`dataset2.py`](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/src/autoencoder2/dataset2.py), but designed for only CPU.

* [`logger.py`](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/src/autoencoder2/logger.py). This file allows to create log files (nothing relevant for the training itself).
