#!/bin/sh
# Node resource configurations
#SBATCH --partition=v100_single_gpu
#SBATCH --nodes=1                               
#SBATCH --ntasks=1                              
#SBATCH --gpus-per-node=1
#SBATCH --error=/scratch/user/OUTPUTS/logs/%A.err
#SBATCH --output=/scratch/user/OUTPUTS/logs/%A.out

# We want names of master node
MASTER=`/bin/hostname -s`
MPORT=$(shuf -i 6000-9999 -n 1)
echo "Master: $MASTER"
echo "Port: $MPORT"

jobdir="$(dirname "$(dirname "$(pwd)")")";

CONFIG=$1
DATASET=$2
WEIGHT=$3
WEIGHT_PATH=${WEIGHT}
SEED=${4:-42}

cd $jobdir;

python eval_linear_svm_ood.py \
--world-size 1 --rank 0 --gpu 0 \
--quiet --sub_dir 'svm' \
--db ${DATASET} \
--config-file ${CONFIG} \
--weight_path ${WEIGHT_PATH} \
--seed $SEED