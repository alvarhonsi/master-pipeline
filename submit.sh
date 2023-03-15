#!/bin/bash
#SRUN --job-name=tak011        # create a short name for your job
#SRUN --nodes=1                # node count
#SRUN --ntasks=1               # total number of tasks across all nodes
#SRUN --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SRUN --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SRUN --gres=gpu:1             # number of gpus per node
#SRUN --time=00:20:00          # total run time limit (HH:MM:SS)
#SRUN --mail-type=begin        # send email when job begins
#SRUN --mail-type=end          # send email when job ends
#SRUN --mail-user=tak011@uib.no # email address
#SRUN --output=slurm-%j.out    # %j is the jobid
#SRUN --export=ALL

conda activate master

echo "Running on $SLURM_JOB_NODELIST"

python run-pipeline.py --dir /experiments/model-test/ --generate --train --eval

echo "Done"