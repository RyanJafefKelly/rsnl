#!/bin/bash -l
max=200
for i in `seq 0 $max`
do
	qsub -V -v "seed=$i" scripts/pbs_jobs/contaminated_slcp.sh
done
