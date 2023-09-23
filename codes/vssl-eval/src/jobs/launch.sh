#!/bin/bash

TYPE=$1
CONFIG=$2
DATASET=$3
WEIGHT=$4

# --------------- ZSL
        
if [ ${TYPE} == "zsl-finetune" ]; then

    bash zsl.sh ${CONFIG} ${DATASET} ${WEIGHT}

elif [ ${TYPE} == "superv-k400-zsl" ]; then

    bash superv_zsl.sh ${CONFIG} ${DATASET}

# --------------- FINETUNE-OOD 

elif [ ${TYPE} == "finetune-ood-cego" ]; then
    
    bash finetune-ood-charadesego.sh ${CONFIG} ${DATASET} ${WEIGHT}
    
elif [ ${TYPE} == "finetune-ood-mit-tiny" ]; then

    bash finetune-ood-mit-tiny.sh ${CONFIG} ${DATASET} ${WEIGHT}
        
elif [ ${TYPE} == "finetune-ood" ]; then
    
    bash finetune-ood.sh ${CONFIG} ${DATASET} ${WEIGHT}
            
# --------------- OPENSET

elif [ ${TYPE} == "openset" ]; then

    bash openset.sh ${CONFIG} ${DATASET} ${WEIGHT}

# --------------- LINEAR-OOD 

elif [ ${TYPE} == "linear-ood-cego" ]; then
    
    bash linear-ood-charadesego.sh ${CONFIG} ${DATASET} ${WEIGHT}
    
elif [ ${TYPE} == "linear-ood-mit-tiny" ]; then

    bash linear-ood-mit-tiny.sh ${CONFIG} ${DATASET} ${WEIGHT}
        
elif [ ${TYPE} == "linear-ood" ]; then
    
    bash linear-svm-ood.sh ${CONFIG} ${DATASET} ${WEIGHT}
            
else
    echo "Unkown type: "${TYPE}
fi
