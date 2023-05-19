#!/bin/bash

### Steps of installation of the Vicuna-13B with GPTQ quantization technique ####
# 1. verify the system has a cuda-capable gpu
# 2. remove anything of the previous installation
# 3. download and install the nvidia driver and cuda toolkit
# 4. setup environmental variables
# 5. install cudnn 11.7
# 6. verify the installation
# 7. install conda package management
# 8. create an virtual env
# 9. install pytorch packages
# 10. git clone the vicuna repo
# 11. git clone GPTQ-for-LLaMa
# 12. download the vicuna-13b-GPTQ-4bit-128g from huggingface
###

# 1. verify your gpu is cuda enable check
lspci | grep -i nvidia

# 2. remove previous installation
sudo apt-get purge nvidia*
sudo apt remove nvidia-*
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt-get autoremove && sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*

# update the system
sudo apt-get update && sudo apt-get upgrade -y

# install other import packages
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

# first get the PPA repository driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# 3. install nvidia driver with dependencies
sudo apt install libnvidia-common-515
sudo apt install libnvidia-gl-515
sudo apt install nvidia-driver-515

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update

# install CUDA-11.7
sudo apt install cuda-11-7

# 4. setup your paths
echo 'export PATH=/usr/local/cuda-11.7/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig

# 5. install cuDNN v11.7
CUDNN_TAR_FILE="cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz"
wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.5.0/local_installers/11.7/cudnn-linux-x86_64-8.5.0.96_cuda11-archive.tar.xz
tar -xzvf ${CUDNN_TAR_FILE}

# copy the following files into the cuda toolkit directory.
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-11.7/include
sudo cp -P cuda/lib/libcudnn* /usr/local/cuda-11.7/lib64/
sudo chmod a+r /usr/local/cuda-11.7/lib64/libcudnn*

# 6. verify the installation, check
nvidia-smi
nvcc -V

# 7. install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sha256sum Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# 8. create conda env
conda create -n vicuna python=3.9
conda activate vicuna

# 9. install Pytorch 2.0.1 with cuda 11.7 with some other necessory packages
conda install -c conda-forge tqdm
conda install -c conda-forge huggingface_hub  # might need to login into the huggingface with access token
conda install -c huggingface transformers
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 10. git clone and install the vicuna model
git clone https://github.com/thisserand/FastChat.git
cd FastChat
pip3 install -e .

# 11. git clone and install GPTQ-for-LLaMA
mkdir repositories
cd repositories
git clone https://github.com/oobabooga/GPTQ-for-LLaMa.git -b cuda
cd GPTQ-for-LLaMa
python setup_cuda.py install

# 12. download the GPTQ-quantized Vicuna model from huggingface
# might require the huggingface account access token
cd ../..
python download-model.py anon8231489123/vicuna-13b-GPTQ-4bit-128g
