#!/bin/bash -l
max=20
for i in `seq 0 $max`
do
	qsub -V -v "seed=$i" scripts/pbs_jobs/snl_well_specified_slcp.sh
done
