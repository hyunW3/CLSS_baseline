#!/bin/bash
cd ../
PORT='tcp://127.0.0.1:12345'
GPU=$1
BS=12  # Total 24
SAVEDIR='saved_voc'

shift 1 # remove $1 (GPU)
OPTION=$@ # --set_deterministic
TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='5-3'
EPOCH=60
INIT_LR=0.01
LR=0.001
INIT_POSWEIGHT=2
MEMORY_SIZE=0  # 100 for MiB-M

NAME='MiB'


# check option --onlyIncremental
# and remove it from OPTION
onlyIncremental=false
if [[ $OPTION == *"--onlyIncremental"* ]]; then
    echo "=== onlyIncremental === "
    NAME=${NAME}
    OPTION=${OPTION/--onlyIncremental/}
    onlyIncremental=true
fi

# check onlyIncremental is True
if [ $onlyIncremental = false ]; then
    python train_voc.py -c configs/config_voc_MiB.json \
    -d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
    --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --pos_weight ${INIT_POSWEIGHT}
fi
python train_voc.py -c configs/config_voc_MiB.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_MiB.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_MiB.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_MiB.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_MiB.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python eval_voc.py -d 0 --test -r ${SAVEDIR}/models/${TASKSETTING}_${TASKNAME}_${NAME}/step_5/checkpoint-epoch${EPOCH}.pth

alert_knock echo "MiB 5-3 done"