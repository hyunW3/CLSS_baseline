#!/bin/bash
# This code is based on CUDA 11.2
# conda create -n DKD python=3.6 -y
# conda install pytorch==1.8.1 torchvision==0.9.1 \
    # torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
# conda install pandas -y 
# conda install matplotlib opencv ipykernel -y 

conda create -n DKD python=3.8 -y
conda install pytorch==1.8.1 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge -y
conda install pandas ipykernel matplotlib -y 
# conda install ipykernel matplotlib opencv -y 
