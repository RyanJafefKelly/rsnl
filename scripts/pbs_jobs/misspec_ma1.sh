#!/bin/bash -l
#PBS -N misspec_ma1
#PBS -l walltime=03:00:00
#PBS -l mem=8GB
#PBS -l ncpus=1
cd $PBS_O_WORKDIR
module load python/3.9.6-gcccore-11.2.0
source .venv/bin/activate
python scripts/run_rsnl_misspec_ma1.py --seed=$seed
deactivate
