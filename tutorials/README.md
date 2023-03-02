# Tutorial

This file shows brief overview of the tutorial to run the code with the different tasks, including a description of the inputs and outputs of each task. First, general aspects of the inputs and outputs are briefly described. Then, a brief description of each individual task is shown, providing the details for each task in separate files. In the folder ["template_scripts"](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/template_scripts/), separate files with scripts for each task are provided, assuming that miniconda is used.

The inputs that require external files, excluding intermmediate processing steps, are:

* **Dataset**. Required for all the methods. The employed dataset must be an H5 file with size *[Nvox,Nmeas]*, being *Nvox* the total of voxels for all the subjects, considering only one MRI volume per subject and excluding the background voxels if possible, and *Nmeas* the total number of measurements or MRI different volumes per subject. For methods optimising the estimation of parameters (CL+par), the number of parameters should be added to number of measurements for *Nmeas*, representing each additional added dimension one quantitative parameter. 

* **Header**. Required for all the methods. In addition to the H5 file, a header csv file is employed. The first and second column represent a voxel number, starting with 0 (for 100000 voxels, values between 0 and 99999, both included), and the third column the ID value (an integer) of each subject, which is repeated the total number of voxels for each specific subject.

* **Acquisition parameters**. Required for all the methods except the pure data-driven selection (CL+NN). This file should be txt file that includes the acquisition parameters, representing each row one specific set of values and each column a specific acquisition parameter (for the MUDI challenge, the three cartesian values of the diffusion gradient orientation, b-value, inversion time and delay time or echo time).

* **Parameters ground truth**. Required for the methods whose loss function depends on the value of the quantitative parameters (CL+par). It is a nifti file with the maps of the parameters used as ground truth to estimate these quantitative parameters.

* **Selected measurements**. Required only for the methods with separate estimation of the MRI signal or quantitative parameters without selection. It is a txt file whose number of lines is equal to the number of sub-selected unique measurements, and each row has the value of the pertinent row from the acquisition parameters file, considering the first row as 0.

* **Mask file** (optional). Only used for the scripts that predict the quantitative parameters and/or the MRI signal after training a particular network. It is a nifti file that contains a mask of the voxels for a specific subject. The number of masked voxels must match the number of voxels per subject in the header file.

Other specific inputs for each task are described in the pertinent files with the detailed description.

The main output files, excluding intermmediate steps, are:

* **MRI signal**. Returned by all scripts used for the final estimation of quantitative parameters and/or prediction of the MRI signal. It contains the predicted MRI signal for a validation subject for all the different acquired volumes (*Nmeas* without maps). If a mask was provided, it is returned as a nifti file. Otherwise, it is returned as a txt file.

* **Quantitative maps**. Returned by all scripts used for the final estimation of quantitative parameters. It contains the quantitative maps for a validation subject. If a mask was provided, it is returned as a nifti file. Otherwise, it is returned as a txt file.

* **Selected measurements**. Returned by the scripts related to training of a selection network. It is a txt file equivalent to the previously defined input related to selected measurements.

The detailed python functions related to the training of the diverse approaches are described in the "[tools](https://github.com/aplanchu/ZEBRA-CA/tree/main/tools/README.md)" folder accesible from the root directory of this project. All the code contained in the scripts available in the "template_scripts" folder should be run in the "tools" folder.
 
The code was designed for a leave-one-out validation approach, so only one validation subject can be considered at the moment. Moreover, the code was designed to be run using one GPU.

The tutorials describing each task are:

* [**Tutorial CL+NN**](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/tutorial_clnn.md) describes the extraction of the sub-selected protocol and the prediction of the whole dataset with the upsampled MRI volumes with CL+NN, i.e., the full data-driven method.

* [**Tutorial CL+eq**](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/tutorial_cleq.md) describes the extraction of the sub-selected protocol with CL+eq, i.e., the physics-informed method.

* [**Tutorial CL+par**](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/tutorial_clpar.md) describes the extraction of the subselected protocol with CL+par, i.e., the method using the quantitative parameters in the loss function.

* [**Tutorial CL+eq estimation**](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/tutorial_cleq_onlyest.md) describes the estimation of the quantitative parameters and the prediction of the MRI signal from a specific sub-selected acquisition protocol optimising the signal prediction.

* [**Tutorial CL+par estimation**](https://github.com/aplanchu/ZEBRA-CA/tree/main/tutorials/tutorial_clpar_onlyest.md) describes the estimation of the quantitative parameters and the prediction of the MRI signal from a specific sub-selected acquisition protocol optimising the parameter estimation.
