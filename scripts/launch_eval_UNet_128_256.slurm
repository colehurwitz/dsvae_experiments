#!/bin/bash
###  replace the .py file in the horovodrun call with your code and it's parameters
###  submit using the command 'sbatch multi_torch.slurm'
###  check queuue using command 'squeue'
###  cancel jobs using command 'scancel <jobnum>'
###  outputs are in  multi_torch_<jobnum>.out and multi_torch_<jobnum>.err

#SBATCH -J eval_unet_128
#SBATCH -o eval_unet_128_%j.out
#SBATCH -e eval_unet_128_%j.err

#SBATCH --mail-user=churwitz@mit.edu
#SBATCH --mail-type=ALL
###  the following parameters will get you two nodes with 4 V100's each
###  note the run time limit is currently only 12 hours, it will be increased
###  next week
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=500g
#SBATCH --time=24:00:00
#SBATCH -p sched_system_all 

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=ml_gpu
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes 
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " NGPUs per node:= " $SLURM_GPUS_PER_NODE 
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes and GPU devices"
echo ""
echo " Run started at:- "
date


## Horovod execution
###  replace the .py with your code and it's parameters
###  submit using the command 'sbatch multi_torch.slurm'
###  check queuue using command 'squeue'
###  cancel jobs using command 'scancel <jobnum>'
###  outputs are in  multi_torch_<jobnum>.out and multi_torch_<jobnum>.err 

python eval_unet_128_256.py \
--train_dir /data/ImageNet/ILSVRC2012/train/ --valid_dir /data/ImageNet/ILSVRC2012/val/ \
--batch_size 100 --image_size 256 \
--output_dir $HOME2/UNET/results/unet_128_256_eval_1/ \
--project_name unet_full_imagenet_128_256_eval \
--model_128_weights /nobackup/users/churwitz/UNET/results/mask_64_128/UNET_pixel_model_128_itr209500.pth \
--model_256_weights /nobackup/users/churwitz/UNET/results/mask_128_256/UNET_pixel_model_256_itr71500.pth

echo "Run completed at:- "
date
