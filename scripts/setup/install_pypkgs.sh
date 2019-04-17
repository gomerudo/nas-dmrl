#!/bin/bash

CURRENT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SET_GLOBALVARS_PATH=${CURRENT_SCRIPT_DIR}/set_globalvars.sh

# Load the global vars
source ${SET_GLOBALVARS_PATH}
# Load the conda script
source ${SETUP_CONDA_PATH}
# Source the environment
source activate ${VENV_NAME}

which conda
which pip
which python
# install the packages
# conda install gym
# conda install pyyaml


