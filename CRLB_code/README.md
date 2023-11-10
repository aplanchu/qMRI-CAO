# Implementation of the Cramér-Rao Lower Bound (CRLB) optimisation
This folder contains the MATLAB code to obtain the acquisition protocol from the CRLB-based optimisation and the selection of the closest measurements from a prespecified acquisition protocol. This code is adapted to the MUDI acquisition protocol and the files must be run once for each desired number of selected measurements. The files are the following:

* [*T1T2sD_CRLB.m*](https://github.com/aplanchu/ZEBRA-CA/tree/main/CRLB_code/T1T2sD_CRLB.m). Implementation of the CRLB-based optimisation. It saves the output protocol from the CRLB procedure. This script must be run in first place. Before running this file, read the documentation of the first lines as some parameters can be modified according to the user's interests.

* [*subselect_CRLB.m*](https://github.com/aplanchu/ZEBRA-CA/tree/main/CRLB_code/subselect_CRLB.m). Code to select the closest measurements of a specific acquisition protocol from a CRLB-based acquisition protocol. This script must be run in last place, after obtaining the CRLB-based acquisition protocol.

* *[*munkres.m*](https://github.com/aplanchu/ZEBRA-CA/tree/main/CRLB_code/munkres.m). Implementation of the Hungarian algorithm. This script is used in [*subselect_CRLB.m*](https://github.com/aplanchu/ZEBRA-CA/tree/main/CRLB_code/subselect_CRLB.m). It requires no independent running from the previous scripts.
