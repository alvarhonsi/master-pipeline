#!/bin/bash
#SBATCH --job-name=tak011        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=tak011@uib.no # email address
#SBATCH --output=./logs/slurm-%j.out    # %j is the jobid
#SBATCH --export=ALL

echo "Running on $SLURM_JOB_NODELIST"

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate master

python run-pipeline.py --dir ./experiments/model-test/ --train --eval -p one-layer-sum

echo "Done"
