#!/bin/bash -l
#PBS -N contaminated_normal
#PBS -l walltime=12:00:00
#PBS -l mem=8GB
#PBS -l ncpus=1
cd $PBS_O_WORKDIR
module load python/3.9.6-gcccore-11.2.0
source .venv/bin/activate
python scripts/run_snl_well_specified_normal.py --seed=$seed
deactivate
