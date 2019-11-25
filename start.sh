#!/usr/bin/bash

#MSUB -l nodes=1:ppn=20
#MSUB -l walltime=60:00:00

cd /work/ws/nemo/hd_wv385-quantum-0/simulations/exponential
singularity exec --app visionary-wafer /work/ws/nemo/hd_wv385-agnes-0/2019-02-27_1.img python generate_sims.py $NUM_PARAMS
