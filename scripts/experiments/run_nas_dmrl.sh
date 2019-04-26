#!/bin/bash

CURRENT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SET_GLOBALVARS_PATH=${CURRENT_SCRIPT_DIR}/../setup/set_globalvars.sh

# Load the global vars
source ${SET_GLOBALVARS_PATH}
# Load the conda script
source ${SETUP_CONDA_PATH}
# Source the environment
conda activate ${VENV_NAME}

TIMESTAMP=`date "+%Y%m%d-%H%M%S"`
SAVE_DIR=${OPENAI_LOGDIR}/models

if [ ! -d ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
fi

echo "save_path directory is: ${SAVE_DIR}"

# Run the baseline
pushd ${OPENAI_BASELINES_PATH}

mkdir nas-env
cp -R ${GIT_STORAGE}/nas-env/resources nas-env/.

git checkout experiments
time python -m baselines.run \
--alg=meta_a2c \
--env=NAS_cifar10-v1 \
--network=meta_lstm \
--save_path=${SAVE_DIR}/meta_a2c_final.model \
--n_tasks=1 \
--num_timesteps=1e3
popd

conda deactivate