#!/bin/bash

################################################################################
## Load the conda.sh script needed to use miniconda at runtime.               ##
##                                                                            ##
## The next environment variables can be set before running this script:      ##
##  - MINICONDA_PATH (Default is: $HOME/workspace/miniconda3)                 ##
##  - VENV_NAME (Default is: nasdmrl)                                         ##
##                                                                            ##
## To enable miniconda loading make sure the next env. vars are set:          ##
##  - WITH_CONDA = YES                                                        ##
################################################################################

# If the variable is set and it is yes
if [ -n ${WITH_CONDA} ] && [ "${WITH_CONDA}" == "YES" ]; then
    
    if [ -z ${MINICONDA_PATH} ]; then
        MINICONDA_PATH=${HOME}/workspace/miniconda3
        if [ -z ${VENV_NAME} ]; then
            VENV_NAME=nasdmrl
        fi
    fi
fi

CONDA_SH=${MINICONDA_PATH}/etc/profile.d/conda.sh
if [ -f ${CONDA_SH} ]; then
    source ${CONDA_SH}
    echo "Successfully loaded: conda"
else
    echo "Error loading conda loader: ${CONDA_SH}"
fi
