#!/bin/bash
cd ../
PORT=$((9000 + RANDOM % 1000))
PORT='tcp://127.0.0.1:'$PORT
GPU=$1 #2,3
BS=12  # Total 24
SAVEDIR='saved_voc'

shift 1 # remove $1 (GPU)
OPTION=$@ # --set_deterministic
TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='5-3'
EPOCH=30
INIT_LR=0.01
LR=0.001
MEMORY_SIZE=0  # 100 for DKD-M
NAME='PLOP'

# check deterministic string exist in OPTION 
if [[ $OPTION == *"--set_deterministic"* ]]; then
    echo "deterministic version"
    NAME=$NAME'_deterministic'
fi
if [[ $OPTION == *"--boost_lr"* ]]; then
    echo "=== boost lr === "
    NAME=${NAME}_boost_lr
fi
# check MEMORY_SIZE == 0 then add into NAME
if [ $MEMORY_SIZE -ne 0 ]; then
    NAME=${NAME}_M${MEMORY_SIZE}
fi


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
    python train_voc.py -c configs/config_voc_PLOP.json \
    -d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
    --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} 
fi
python train_voc.py -c configs/config_voc_PLOP.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_PLOP.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_PLOP.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_PLOP.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_PLOP.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python eval_voc.py -d 0 --test -r ${SAVEDIR}/${TASKSETTING}_${TASKNAME}_${NAME}/step_5/checkpoint-epoch${EPOCH}.pth

echo "voc 5-3 PLOP done"