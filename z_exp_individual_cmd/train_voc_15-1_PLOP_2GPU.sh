#!/bin/bash
cd ../
START_DATE=$(date '+%Y-%m-%d')

PORT=$((9000 + RANDOM % 1000))
PORT='tcp://127.0.0.1:'$PORT
GPU=0,1
BS=8  # Total 24
SAVEDIR='saved_voc'

TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='15-1'
EPOCH=30
INIT_LR=0.01
LR=0.001
OPTION=$@ # --set_deterministic
# INIT_POSWEIGHT=2
MEMORY_SIZE=0  # 100 for DKD-M
NAME='PLOP'
# check deterministic string exist in OPTION 
if [[ $OPTION == *"--set_deterministic"* ]]; then
    echo "deterministic version"
    NAME=$NAME'_deterministic'
else
    echo "non-deterministic version"
    NAME=$NAME'_non_deterministic'
fi
if [[ $OPTION == *"--boost_lr"* ]]; then
    echo "=== boost lr === "
    NAME=${NAME}_boost_lr
fi
# check MEMORY_SIZE == 0 then add into NAME
if [ $MEMORY_SIZE -ne 0 ]; then
    NAME=${NAME}_M${MEMORY_SIZE}
fi
# if `find "./saved_voc/models/overlap_15-1_"$NAME"/" -name "*.pth"` ; then
#     echo "Found saved model."
#     echo "==========="
#     echo `find "./saved_voc/models/overlap_15-1_"$NAME -name "*.pth"`
#     echo "==========="
#     echo "Do you want to remove whole folder? [y/n]"
#     read answer
#     if [ $answer = "y" ] ; then
#         rm -r "./saved_voc/models/overlap_15-1_"$name
#         rm -r "./saved_voc/models/overlap_15-1_"$NAME
#     else 
#         echo "Continue."
#     fi
# else
#     echo "No saved model found."
#     rm -r "./saved_voc/models/overlap_15-1_"$NAME
#     rm -r "./saved_voc/models/overlap_15-1_"$NAME
# fi


# alert_knock python train_voc.py -c configs/config_voc_PLOP.json \
# -d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
# --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} && 
rm -r ./saved_voc/models/overlap_15-1_PLOP_non_deterministic/step_1
rm -r ./saved_voc/log/overlap_15-1_PLOP_non_deterministic/step_1

alert_knock python train_voc.py -c configs/config_voc_PLOP.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE} 

python train_voc.py -c configs/config_voc_PLOP.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_PLOP.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_PLOP.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_PLOP.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python eval_voc.py -d 0 \
    -r ${SAVEDIR}/models/${TASKSETTING}_${TASKNAME}_${NAME}/step_5/checkpoint-epoch${EPOCH}.pth 

alert_knock echo "PLOP in DKD finished."
