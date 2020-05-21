# Predicting Atomization Energy of Molecules with Machine Learning

This repository contains scripts and code related to predicting high-accuracy atomization energies of molecules with machine learning.
Specifically, we seek to predict the atomization energy predicted using G4MP2.
In particular, we include comparisons between machine learning methods and the validation of the ability of these approaches
to predict the properties of molecules with sizes larger than the those in the training set.

We are gradually increasing the diversity of information we can use as inputs for the ML models.
For now, all of our molecules require the relaxed geometry of the molecule using B3LYP/6-31G(2df,p). 
and the total energy from either B3LYP or &omega;B97xd for that geometry. 
We also have models trained using different machine learning approaches, such as [FCHL](https://aip.scitation.org/doi/full/10.1063/1.5020710) and [SchNet](https://aip.scitation.org/doi/10.1063/1.5019779), that each have
evaluation rate and accuracy tradeoffs.

## Using this Repository

The repository records the full proveance of our machien elarning models. 
You can gain access to the models by either installing and re-running the notebooks or
invoking pre-trained models hosted on [DLHub](https://www.dlhub.org/).

### Installation

The scripts and notebooks in this project require the utility scripts in [`jcesr_ml`](jcesr_ml) and the requirements
are listed in the `environment.yml` file.

Install the environment with [Anaconda](https://conda.io/en/latest/) by calling:

```bash

conda env create --file environment.yml
```

The project is broken into directories that focus on specific issues (e.g., predicting the atomization energy 
without requiring a DFT geometry). 
Each directory contains a series of notebooks that are labeled in the order in which they should be executed.
The notebooks should describe their purpose and results.


A copy of this repository with the output files (which are too large to host on GitHub) from the notebooks
and scripts is on the Materials Data Facility: [link](http://dx.doi.org/doi:10.18126/M2V65Z).
**NOTE**: We will be releasing an updated version with the results from later work.


### Executing Models from DLHub

You can use DLHub to run our machine learning models without needing to install this repository.
The notebooks in the [`dlhub`](./dlhub) folder illustrate how to use DLHub to run the machine 
learning models on remote services. 
You only need install the DLHub SDK (`pip install dlhub_sdk`) to use the webhosted models.


## Citation

The papers describing this work include:

- Logan Ward, Ben Blaiszik, Ian Foster, Rajeev S. Assary, Badri Narayanan, and Larry Curtiss. "Machine Learning Prediction of Accurate Atomization Energies of Organic Molecules from Low-Fidelity Quantum Chemical Calculations." MRS Communications (2019), 891. [doi:10.1557/mrc.2019.107](https://doi.org/10.1557/mrc.2019.107).
- Naveen Dandu, Logan Ward, Rajeev S. Assary, Paul Redfern, Badria Narayanan, Ian Foster, Larry Curtiss. "Quantum Chemically Informed Machine Learning: Prediction of Energies of Organic Molecules with 10 to 14 Non-Hydrogen Atoms." _in revision_


Please consult these references for more information about our work and cite them if you use this repository. 
