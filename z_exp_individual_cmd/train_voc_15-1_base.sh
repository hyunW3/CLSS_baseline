#!/bin/bash
cd ../
PORT=$((9000 + RANDOM % 1000))
PORT='tcp://127.0.0.1:'$PORT
GPU=$1 #0,1,2,3
BS=12  # Total 24
SAVEDIR='saved_voc'

shift 1
OPTION=$@ # --set_deterministic
echo $OPTION # --onlyBase, --onlyIncremental, --use_cosine, --OCFM, --main_loss CE, --main_loss MBCE
TASKSETTING='overlap'  # or 'disjoint'
TASKNAME='15-1'
EPOCH=60
INIT_LR=0.001
LR=0.0001
INIT_POSWEIGHT=2
MEMORY_SIZE=0  # 100 for DKD-M

NAME="base"
# check deterministic string exist in OPTION 
if [[ $OPTION == *"--set_deterministic"* ]]; then
    echo "deterministic version"
    NAME=$NAME'_deterministic'
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
    OPTION=${OPTION/--onlyIncremental/}
    onlyIncremental=true
fi
if [[ $OPTION == *"--onlyBase"* ]]; then
    echo "=== onlyBase === "
    OPTION=${OPTION/--onlyBase/}
    onlyBase=true
    NAME=${NAME}_onlyBase
fi

if [[ $OPTION == *"--use_cosine"* ]]; then
    echo "=== use_cosine === "
    NAME=${NAME}_cosine_clf
fi
# base + alpha
if [[ $OPTION == *"--OCFM"* ]]; then
    echo "=== OCFM === "
    NAME=${NAME}_OCFM
fi
# do not want to include WBCE
if [[ $OPTION == *"--main_loss CE"* ]]; then
    echo "=== CE === "
    NAME=${NAME}_CE
fi
if [[ $OPTION == *"--main_loss MBCE"* ]]; then
    echo "=== MBCE === "
    NAME=${NAME}_MBCE
fi
if [[ $OPTION == *"--main_loss BCE"* ]]; then
    echo "=== BCE === "
    NAME=${NAME}_BCE
fi

# only base method, it is finetune (not continue training)
# check onlyIncremental is True
if [ $onlyIncremental = false ]; then
    alert_knock python train_voc.py -c configs/config_voc_base.json \
    -d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
    --task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS} --pos_weight ${INIT_POSWEIGHT}
fi
if [ $onlyBase = true ]; then
    exit
fi
# if file exists
if [ -f ${SAVEDIR}/${TASKSETTING}_${TASKNAME}_${NAME}/step_0/checkpoint-epoch${EPOCH}.pth ]; then
    echo "file exists"
else 
    echo "file not found, end script"
    alert_knock echo "voc 15-1 "$NAME" not done (step0 not finished)"
    exit
fi
alert_knock python train_voc.py -c configs/config_voc_base.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step  1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}    

python train_voc.py -c configs/config_voc_base.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_base.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python train_voc.py -c configs/config_voc_base.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

alert_knock python train_voc.py -c configs/config_voc_base.json \
-d ${GPU} --multiprocessing_distributed --dist_url ${PORT} --save_dir ${SAVEDIR} --name ${NAME} ${OPTION} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

python eval_voc.py -d 0 --test \
    -r ${SAVEDIR}/${TASKSETTING}_${TASKNAME}_${NAME}/step_5/checkpoint-epoch${EPOCH}.pth # saved_voc/models/overlap_15-1_DKD/step_5/checkpoint-epoch60.pth

alert_knock echo "voc 15-1 "$NAME" done"