#a simple sample

#!/bin/bash

echo "#!/bin/bash" > script.slurm

echo "#SBATCH --job-name=kevin_job" >> script.slurm # the name of your job

echo "#SBATCH --output=result.out" >> script.slurm # a file for recording the print out of your experiment

echo "#SBATCH --nodes=1 " >> script.slurm # request n nodes

echo "#SBATCH --ntasks-per-node=1" >> script.slurm # total number of times running main.py

echo "#SBATCH --cpus-per-task=42" >> script.slurm # each GPU contain 128 cpus (workers), each GPU is a task.

echo "#SBATCH --mem-per-cpu=3850" >> script.slurm # required memory per cpu

echo "#SBATCH --gres=gpu:ampere_a100:1" >> script.slurm # maximum 3 GPUs per node.

echo "#SBATCH --partition=gpu" >> script.slurm

echo "#SBATCH --time=00:30:00" >> script.slurm # maximum 48 hours of consecutive running

echo "#SBATCH --account=su004-neuralnet" >> script.slurm #budget account of leicester university

echo "source ~/.bashrc" >> script.slurm

echo "conda activate kevin_test" >> script.slurm

echo "cd /home/x/xk18/PhD_code_exp/project_3" >> script.slurm

echo "srun python simple_test.py " >> script.slurm

sbatch script.slurm # launch the job

rm script.slurm # remove script.slurm
