#!/bin/bash

##############################################################################################################
##################################### UPDATE BELOW VALUES BEFORE RUNNING #####################################
##############################################################################################################

# Provide the name of the conda virtual environment
THESIS_VENV=intent                                                              # e.g. intent/thesis/...

##############################################################################################################
##################################### DO NOT MODIFY THE SETTINGS BELOW #######################################
##############################################################################################################

# Cleanup environment
echo "Cleaning up environment..."
echo ""
conda remove --name $THESIS_VENV --all --yes && echo "Enironment cleanup successful!"

# Activate conda virtual environment to run from bash
echo "Activating conda virtual environment to run from bash..."
echo ""
conda init bash

# Create a conda virtual environment
echo "Creating a conda virtual environment..."
echo ""
source ~/Data/miniconda3/etc/profile.d/conda.sh                # Workaround (conda activate does not work from bash)
conda create --yes --name $THESIS_VENV python=3.8 && echo "Anaconda virtual environment created successfully!"
conda activate $THESIS_VENV && echo "Anaconda virtual environment activated successfully!"

# Install all dependecies
echo "Installing all dependecies..."
echo ""
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y && echo "Pytorch 1.8.0 installed successfully with CUDA 11.1!"
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric
pip install torch-geometric-temporal

# Deactivate Conda Environment
echo "Deactivating Conda Environment..."
echo ""
conda deactivate
return 0
