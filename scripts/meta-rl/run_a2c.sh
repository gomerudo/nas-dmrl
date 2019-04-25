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
git checkout master
time python -m baselines.run \
--alg=a2c \
--env=AssaultNoFrameskip-v0 \
--network=lstm \
--save_path=${SAVE_DIR}/a2c.model \
--num_timesteps=1e6
popd

conda deactivate