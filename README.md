# rsnl
Library to run RSNL to obtain results in the "Misspecifcation-robust Sequential Neural Likelihood for Simulation-based Inference" neurips submission

The _rsnl_ package is structured as follows:
notebooks: contains jupyter notebooks to plot results

res/ - contains results saved from inference runs

rsnl/ - contains logic for RSNL algorithm and inference

rsnl/examples/ - code for the four inference tasks

scripts/ - python and pbs scripts to run full inference tasks

scripts in the base folder \*.sh are used to submit multiple PBS jobs for an inference task with different seeds

To install dependencies run: _make install_

