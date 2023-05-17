#!/bin/bash -l
max=150
for i in `seq 0 $max`
do
	qsub -V -v "seed=$(($i+50))" scripts/pbs_jobs/misspec_ma1.sh
done
