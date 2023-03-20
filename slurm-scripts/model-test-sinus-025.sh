#!/bin/bash
#SBATCH --job-name=tak011        # create a short name for your job
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=06:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=tak011@uib.no # email address
#SBATCH --output=./logs/slurm-%j.out    # %j is the jobid
#SBATCH --export=ALL

echo "Running on $SLURM_JOB_NODELIST"

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate master

python run-pipeline.py --dir ./experiments/model-tests/sinusmodel-sigma025 --generate --train --eval --dev cuda

echo "Done"
