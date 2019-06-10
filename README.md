# Predicting Atomization Energy of Molecules with Machine Learning

This repository contains scripts and code related to predicting the atomization energy of molecules with machine learning.
In particular, we include comparisons between machine learning methods and the validation of the ability of these approaches
to predict the properties of molecules with sizes larger than the those in the training set.

The paper describing this activity is in preparation for submission to MRS Communications.

A copy of this repository with the generated data files (which are too large to host on GitHub)
will soon be able on the Materials Data Facility.
It will also soon be possible to run all of these scripts in a pre-configured Virtual Machine via [WholeTale](http://wholetale.org)
and to excecute the models via [DLHub](https://dlhub.org)

# Installation

The scripts in this project require the utility scripts in [`jcesr_ml`](jcesr_ml) and the requirements
are listed in the `environment.yml` file.

Install the environment with [Anaconda](https://conda.io/en/latest/) by calling:

```bash

conda env create --file environment.yml
```
