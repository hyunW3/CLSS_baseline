# CLSS (Contiunal Learning Semantic Segmentation) baselines
- This is CLSS baseline implementation
- Currently, it includes MiB, PLOP, DKD.
- The main implementation is based on dataset VOC2012 and ResNet-101 backbone.
- ADE 20K dataset is not supported currently, but will be supported soon. 

## Environment
- CUDA 11.1
- python 3.8
- The shell code for setting environment is in `scripts/env_create.sh`

## training
- The shell code for training is in `./z_exp_individual_cmd`
```
    ./train_{dataset}_{scenarios}_{method}.sh
    For examples, ./train_voc_10-1_DKD.sh
```
- The configuration for each method is in 'configs/config_{dataset}_{method}.yaml'
- The results are updated in wandb 


## Acknowledgements
* This code is based on DKD (https://github.com/cvlab-yonsei/DKD#decomposed-knowledge-distillation-for-class-incremental-semantic-segmentation) codespaces.
* All implementations have been borrowed from existing code implementations (MiB, DKD, PLOP)
