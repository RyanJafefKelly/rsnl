#!/bin/bash -l
#PBS -N snl_SIR
#PBS -l walltime=72:00:00
#PBS -l mem=16GB
#PBS -l ncpus=1
#PBS -l cputype=6140
cd $PBS_O_WORKDIR
module load python/3.9.6-gcccore-11.2.0
source .venv/bin/activate
python scripts/run_snl_sir.py  --seed=$seed
deactivate
