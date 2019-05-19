#!/bin/bash

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

# Not available in conda, use pip
pip install gym
pip install opencv-python
pip install atari_py
pip install gin-config

conda deactivate