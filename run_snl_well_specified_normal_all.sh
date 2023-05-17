#!/bin/bash -l
max=200
for i in `seq 0 $max`
do
	qsub -V -v "seed=$i" scripts/pbs_jobs/snl_well_specified_normal.sh
done
