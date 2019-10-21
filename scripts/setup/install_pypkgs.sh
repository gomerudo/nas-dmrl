#!/bin/bash

################################################################################
## Install the required packages for all the scripts/experiments.             ##
##                                                                            ##
## Assumptions of this script are:                                            ##
##  - Work within a conda environment                                         ##
##  - Tensorflow version: 1.12                                                ##
##  - CUDA version: 9                                                         ##
##                                                                            ##
## All the env. vars and options from scripts/setup/set_globalvars.sh and     ##
## scripts/setup/setup_condaenv.sh apply.                                     ##
################################################################################

CURRENT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SET_GLOBALVARS_PATH=${CURRENT_SCRIPT_DIR}/set_globalvars.sh

# Load the global vars
source ${SET_GLOBALVARS_PATH}
# Load the conda script
source ${SETUP_CONDA_PATH}
# Source the environment
conda activate ${VENV_NAME}

############################# INSTALL THE PACKAGES #############################

# Available in conda
conda install joblib
conda install pyyaml
conda install pandas
conda install tensorflow-gpu=1.12.0
conda install cudatoolkit=9.0  # Immediately downgrade
conda install cloudpickle
conda install Pillow=5.4.1
conda install scikit-learn

# Not available in conda, use pip
pip install gym
pip install opencv-python
pip install atari_py
pip install gin-config

conda deactivate