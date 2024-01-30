#!/bin/sh
# Node resource configurations
#SBATCH --partition=v100_full_node
#SBATCH --nodes=2                               
#SBATCH --ntasks=2                             
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
log_path="/scratch/user/OUTPUTS/logs"

CONFIG=$1
DB=${2:-"kinetics400"}
RESUME=$3


if [ -z $RESUME ];
then
    echo "

    cd $jobdir;
    python main_byol.py --server 'server-name' \
                --dist-url tcp://${MASTER}:${MPORT} \
                --dist-backend 'nccl' \
                --multiprocessing-distributed \
                --world-size ${SLURM_JOB_NUM_NODES} --rank \${SLURM_PROCID} \
                --job_id ${SLURM_ARRAY_JOB_ID}"_"${DB} \
                --quiet --sub_dir 'pretext' \
                --config-file ${CONFIG} \
                --seed 42 \
                --db ${DB} |& tee ${log_path}/${SLURM_ARRAY_JOB_ID}_\${SLURM_PROCID}.out
                
    if (( ${SLURM_PROCID} != 0 ));then         
        while [ ! -f "${log_path}/${SLURM_JOBID}_0.out" ]
            do
            sleep 2
            echo 'waiting for rank 0'
        done
    fi
                
                " > srun_worker_${SLURM_ARRAY_JOB_ID}_${SLURM_PROCID}.sh
    srun bash srun_worker_${SLURM_ARRAY_JOB_ID}_${SLURM_PROCID}.sh
else
    echo "
    
    cd $jobdir;
    python main_byol.py --server 'server-name' \
                --dist-url tcp://${MASTER}:${MPORT} \
                --dist-backend 'nccl' \
                --multiprocessing-distributed \
                --world-size ${SLURM_JOB_NUM_NODES} --rank \${SLURM_PROCID} \
                --job_id ${SLURM_ARRAY_JOB_ID}"_"${DB} \
                --quiet --sub_dir 'pretext' \
                --config-file ${CONFIG} --resume ${RESUME} \
                --seed 42 \
                --db ${DB}  |& tee ${log_path}/${SLURM_ARRAY_JOB_ID}_\${SLURM_PROCID}.out
                
    if (( ${SLURM_PROCID} != 0 ));then         
        while [ ! -f "${log_path}/${SLURM_JOBID}_0.out" ]
            do
            sleep 2
            echo 'waiting for rank 0'
        done
    fi
                
                " > srun_worker_${SLURM_ARRAY_JOB_ID}_${SLURM_PROCID}.sh
    srun bash srun_worker_${SLURM_ARRAY_JOB_ID}_${SLURM_PROCID}.sh
fi
