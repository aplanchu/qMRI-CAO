# Physics-informed concrete autoencoder
This project contains the code for the following tasks:

* Full data-driven optimal selection of a subset of MRI volumes and prediction of the MRI signal of the whole dataset using a concrete autoencoder. This method is abbreviated as **CL+NN** (concrete layer + neural network).

* Physics-informed selection of a subset of MRI volumes using a concrete autoencoder predicting the MRI signal of the whole dataset through the estimation of quantitative MRI parameters based on the ZEBRA model. The loss function used the prediction of the MRI signal. This method is abbreviated as **CL+eq** (concrete layer + equation).

* Physics-informed selection of a subset of MRI volumes using a concrete autoencoder predicting the MRI signal of the whole dataset through the estimation of quantitative MRI parameters based on the ZEBRA model. The loss function used the estimation of the quantitative parameters, in contrast to CL+eq. The estimation of the parameters is data-driven, but the previous extraction of the *ground truth* maps and the MRI signal are model-based. This method is abbreviated as **CL+par** (concrete layer + parameters).

* The separate estimation of the quantitative parameters and prediction of the MRI signal of a whole dataset from a specific reduced subset of MRI volumes optimising the signal prediction (CL+eq) or the parameter estimation (CL+par) based on the ZEBRA model.

The idea of the three described joint selection and prediction methods is to extract the optimal subset of MRI volumes with a user-defined size (number of sub-measurements) using the information of the predicted signal or estimated quantitative parameters.

The architecture of the three approaches is sketched below, including the equation employed for both physics-informed methods:

<img src="https://github.com/aplanchu/ZEBRA-CA/blob/main/selection_methods.png" width="1024">

This figure shows an exampled based on the data employed in the Multi-dimensional Diffusion MRI (MUDI) challenge. From out of a total of 1344 MRI volumes, 500 sub-measurements were sampled and employed to predict the MRI signal from the whole dataset via a neural network (CL+NN) or the equation in the red area (CL+eq, and CL+par only after training is finished).

## Dependencies

It is recommended to use Conda or Miniconda. An environment file is provided with the necessary dependencies. The environment can be installed as follows:

```
$ conda env create -f ZEBRA_MUDIenv.yml
```

Or to update a specific environment:

```
$ conda env update -n my_environment -f ZEBRA_MUDIenv.yml
```

The project can be installed as follows:

```
$ python -m pip install -e .
```

## User guide and tutorials

The code is prepared to run training separately from the final prediction of the MRI signal and the estimations of the quantitative parameters for CL+eq and CL+par. Any script must be run at the path where the folders in the "tools" directory are stored. The detailed description of the different tasks of the project is available in the "[tutorials](https://github.com/aplanchu/ZEBRA-CA/tutorials/README.md)" folder. More detailed information about the python files with the code used to train the networks and estimate the quantitative paramters or predict the MRI signal is available in the "[tools](https://github.com/aplanchu/ZEBRA-CA/tools/README.md)" folder.

## License

This project is distributed under the MIT License, Copyright (c) 2023 Álvaro Planchuelo-Gómez. For further details, read the [license](https://github.com/aplanchu/ZEBRA-CA/LICENSE) file.

## Citation and References

The preliminary results related to this project have been accepted for publication in the 2023 International Society for Magnetic Resonance in Medicine Annual Meeting & Exhibition:

Planchuelo-Gómez Á, Descoteaux M, Aja-Fernández S, Hutter J, Jones DK, Tax CMW. "Comparison of data-driven and physics-informed approaches for optimising multi-contrast MRI acquisition protocols". Proceedings of the 2023 International Society for Magnetic Resonance in Medicine Annual Meeting & Exhibition (accepted for publication).

The results related to the first full data-driven concrete autoencoders approach applied to MRI data are in the following reference:

Tax CMW, Larochelle H, De Almeida Martins JP, Hutter J, Jones DK, Chamberland M, Descoteaux M. "Optimising multi-contrast MRI experiment design using concrete autoencoders". Proceedings of the 2021 International Society for Magnetic Resonance in Medicine Annual Meeting & Exhibition, p. 1240. URL: [https://archive.ismrm.org/2021/1240.html](https://archive.ismrm.org/2021/1240.html).

The references for the signal equation employed in this project are:

Hutter J, Slator PJ, Christiaens D, Teixeira RPA, Roberts T, Jackson L, Price AN, Malik S, Hajnal JV. "Integrated and efficient diffusion-relaxometry using ZEBRA". Scientific reports 2018, 8: 1-13. DOI: [10.1038/s41598-018-33463-2](https://doi.org/10.1038/s41598-018-33463-2).

Tax CM, Kleban E, Chamberland M, Baraković M, Rudrapatna U, Jones DK. "Measuring compartmental T2-orientational dependence in human brain white matter using a tiltable RF coil and diffusion-T2 correlation MRI". NeuroImage 2021, 236: 117967. DOI: [10.1016/j.neuroimage.2021.117967](https://doi.org/10.1016/j.neuroimage.2021.117967).

## Acknowledgements

The dataset of the MUDI challenge employed to develop the code from this project was kindly provided by [Centre for the Developing Brain](http://cmic.cs.ucl.ac.uk/cdmri20/challenge.html). Maarten de Klerk contributed to the organisation of the code for the development of the CL+NN approach.

This project was funded by the Dutch Research Council (NWO) with a Veni Grant (17331) and by the Wellcome Trust with a Sir Henry Wellcome Fellowship (215944/Z/19/Z). ÁPG is currently supported by the European Union (NextGenerationEU).
