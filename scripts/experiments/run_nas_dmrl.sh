#!/bin/bash

CURRENT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SET_GLOBALVARS_PATH=${CURRENT_SCRIPT_DIR}/../setup/set_globalvars.sh

# Load the global vars
source ${SET_GLOBALVARS_PATH}
# Load the conda script
source ${SETUP_CONDA_PATH}
# Source the environment
conda activate ${VENV_NAME}

current_date=`date +%Y%m%d%H%M%S`
LOGDIR="${WORKSPACE}/results/opeanai-${current_date}"
export OPENAI_LOGDIR=${LOGDIR}

SAVE_DIR=${OPENAI_LOGDIR}/models

if [ ! -d ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
fi

# Run the baseline
pushd ${OPENAI_BASELINES_PATH}

if [ ! -d nas-env ]; then
    mkdir nas-env
fi

cp -R ${GIT_STORAGE}/nas-env/resources nas-env/.

if [ ! -d workspace ]; then
    mkdir workspace
fi

git checkout experiments
time python -m baselines.run \
--alg=meta_a2c \
--env=NAS_cifar10-v1 \
--network=meta_lstm \
--save_path=${SAVE_DIR}/meta_a2c_final.model \
--n_tasks=1 \
--num_timesteps=1e3

# Zip the results for easy export
zip -r nasenv_results.zip workspace
popd

conda deactivate

pushd ${OPENAI_LOGDIR}
cd ..
dir_name=`basename ${OPENAI_LOGDIR}`
# Zip the results for easy export
zip -qr ${dir_name}.zip ${dir_name}
popd

