#!/bin/bash -l
#PBS -N SIR
#PBS -l walltime=48:00:00
#PBS -l mem=8GB
#PBS -l ncpus=1
#PBS -l cputype=6140
cd $PBS_O_WORKDIR
module load python/3.9.6-gcccore-11.2.0
source .venv/bin/activate
python scripts/run_rsnl_sir.py  --seed=$seed
deactivate
