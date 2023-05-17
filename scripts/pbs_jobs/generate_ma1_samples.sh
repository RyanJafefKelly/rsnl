#!/bin/bash -l
#PBS -N generate_ma1_samples
#PBS -l walltime=24:00:00
#PBS -l ncpus=8
#PBS -j eo
#PBS -m abe
#PBS -M r21.kelly@hdr.qut.edu.au
cd $PBS_O_WORKDIR
module load python/3.9.6-gcccore-11.2.0
source .venv/bin/activate
python scripts/get_misspec_ma1_abc_samples.py
deactivate
