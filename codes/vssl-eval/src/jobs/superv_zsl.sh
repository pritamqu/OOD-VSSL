#!/bin/sh
# Node resource configurations
#SBATCH --partition=v100_full_node
#SBATCH --nodes=1                               
#SBATCH --ntasks=1                              
#SBATCH --gpus-per-node=4
#SBATCH --error=/scratch/user/OUTPUTS/logs/%A.err
#SBATCH --output=/scratch/user/OUTPUTS/logs/%A.out
#SBATCH --array=0-1%1


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

python eval_supervised_zsl.py \
            --checkpoint_path ${CKPTDIR} \
            --dist-url tcp://${MASTER}:${MPORT} \
            --dist-backend 'nccl' \
            --multiprocessing-distributed \
            --world-size 1 --rank 0 \
            --quiet --sub_dir 'zsl' \
            --db ${DATASET} \
            --config-file ${CONFIG} \
            --seed 42 