#!/bin/bash

CURRENT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SET_GLOBALVARS_PATH=${CURRENT_SCRIPT_DIR}/set_globalvars.sh

# Load the global vars
source ${SET_GLOBALVARS_PATH}
# Load the conda script
source ${SETUP_CONDA_PATH}
# Source the environment
source activate ${VENV_NAME}

TIMESTAMP=`date "+%Y%m%d-%H%M%S"`
SAVE_DIR=${WORKSPACE}/results/a2c-${TIMESTAMP}
if [ ! -d ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
fi

echo "save_path directory is: ${SAVE_DIR}"

# Run the baseline
pushd ${OPENAI_BASELINES_PATH}/baselines
time python -m baselines.run \
--alg=a2c \
--env=PongNoFrameskip-v4 \
--network=lstm \
--save_path=${SAVE_DIR} \
--num_timesteps=1000
popd