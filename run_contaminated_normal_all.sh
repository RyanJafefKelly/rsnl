#!/bin/bash -l
max=100
for i in `seq 0 $max`
do
	qsub -V -v "seed=$(($i+100))" scripts/pbs_jobs/contaminated_normal.sh
done
