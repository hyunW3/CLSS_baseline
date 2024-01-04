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
    For MiB, PLOP which require 2 GPUs, please pass the argument GPU_NUMBER (ex 0,1) to .sh file
```
- The configuration for each method is in 'configs/config_{dataset}_{method}.yaml'
- The results are updated in wandb 


## configuration for each baseline
### Pascal VOC 2012
| coonfig                                  | MiB                                        | PLOP                        | DKD                | STAR                |
|------------------------------------------|--------------------------------------------|-----------------------------|--------------------|---------------------|
| epoch                                    | 30                                         | 30                          | 60                 | 60                  |
| lr                                       | 0.01 / 0.001                               | 0.01 / 0.001                | 0.001/ 0.0001      | 0.001/ 0.0001       |
| $\gamma$ (pos weight for BCE Loss)       | UnCE                                       | 1                           | 2 / 1              | 4                   |
| Optimizer                                | SGD (momentum 0.9, wd 1e-4, nesterov True) | SGD (momentum 0.9, wd 1e-4) | SGD (momentum 0.9) | Adam (momentum 0.9) |
| $\alpha,\beta$ (hyperparameter for loss) | 10 (lkd)                                   | 1 (pod)                     | 5 / 5 (kd / dkd)   | 5 / 0.05 (pkd/cont) |
| batch size                               | 24                                         | 24                          | 32                 | 24                  |
| lr Schedular                             | PolyLR                                     | PolyLR                      | PolyLR             |                     |
| GPUs                                     | RTX titian x 2                             | ? x 2                           | A5000 x 4          | RTX 3090 x 2        |
| augmentation                             | same as [1]                                |

### ADE 20K
| config                                   | MiB                         | PLOP                        | DKD                     | STAR                |
|------------------------------------------|-----------------------------|-----------------------------|-------------------------|---------------------|
| epoch                                    | 60                          | 60                          | 100                     | 100                 |
| lr                                       | 0.01 / 0.001                | 0.01 / 0.001                | 0.0025 / 0.00025        | 0.00025 / 0.000025  |
| $\gamma$ (pos weight for BCE Loss)       | UnCE                        | 1                           | 35                      | 30                  |
| Optimizer                                | SGD (momentum 0.9, wd 1e-4) | SGD (momentum 0.9, wd 1e-4) | SGD (momentum 0.9)      | Adam (momentum 0.9) |
| $\alpha,\beta$ (hyperparameter for loss) | 10 (lkd)                    | 1 (pod)                     | 5 / 5 (kd / dkd)        | 5 / 0.05 (pkd/cont) |
| batch size                               | 24                          | 24                          | 24                      | 24                  |
| lr Schedular                             | PolyLR                      | PolyLR                      | PolyLR + linear warm up |                     |
| GPUs                                     | RTX titian x 2              | ? x 2                       | A5000 x 4               | RTX 3090 x 2        |
| augmentation                             | same as [1]                 |                             |                         |                     |


## Acknowledgements
* This code is based on DKD (https://github.com/cvlab-yonsei/DKD#decomposed-knowledge-distillation-for-class-incremental-semantic-segmentation) codespaces.
* All implementations have been borrowed from existing code implementations (MiB, DKD, PLOP)
