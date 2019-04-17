#!/bin/bash

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
