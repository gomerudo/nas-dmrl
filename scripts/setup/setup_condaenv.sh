#!/bin/bash

if [ -n ${WITH_CONDA} ] && [ "${WITH_CONDA}" == "YES" ]; then
    
    if [ -z ${MINICONDA_PATH} ]; then
        MINICONDA_PATH=${HOME}/workspace/miniconda3
        if [ -z ${VENV_NAME} ]; then
            VENV_NAME=nasdmrl
        fi
    fi

fi

source ${MINICONDA_PATH}/etc/profile.d/conda.sh