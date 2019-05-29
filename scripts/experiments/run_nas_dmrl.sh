#!/bin/bash

# Obtain the PID since the beginning, to help debugging.
MASTER_PID=$$
echo "Master process' PID: ${MASTER_PID}"

# Obtain the script's current directory
CURRENT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Set the global environment variables
SET_GLOBALVARS_PATH=${CURRENT_SCRIPT_DIR}/../setup/set_globalvars.sh
source ${SET_GLOBALVARS_PATH}

# Activate the environment
source ${SETUP_CONDA_PATH}
echo "Activating conda environment ${VENV_NAME}"
conda activate ${VENV_NAME}

# Obtain the current timestep, so that we can create isolated logs
current_date=`date +%Y%m%d%H%M%S`
LOGDIR="${WORKSPACE}/results/opeanai-${current_date}"

# Export the OPENAI_LOGDIR as needed by openai-baselines project
export OPENAI_LOGDIR=${LOGDIR}

# Define the place where we will store the resulting model
SAVE_DIR=${OPENAI_LOGDIR}/models

if [ ! -d ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
fi

# Run the baseline
pushd ${OPENAI_BASELINES_PATH}

if [ ! -d workspace ]; then
    mkdir workspace
fi

git checkout experiments

# Let the nvidia monitoring running in the background for as long as the process runs
echo "Running GPU monitoring tool in the background"
nvidia-smi \
--query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used \
--format=csv \
-l 5 > ${WORKSPACE}/runs/smi-${current_date}.csv 2>&1 &

time python -m baselines.run \
--alg=meta_a2c \
--env=NAS_cifar10-v1 \
--network=meta_lstm \
--save_path=${SAVE_DIR}/meta_a2c_final.model \
--n_tasks=1 \
--nsteps=5 \
--num_timesteps=1e3
#--load_path=/home/TUE/20175601/workspace/results/opeanai-20190528182301/models/meta_a2c_tmp-1.mdl

popd

# Deactivate the environment cause we are done with the python steps
conda deactivate

# Go to the OpenAI logdir's parent and zip the results
pushd ${OPENAI_LOGDIR}/..
dir_name=`basename ${OPENAI_LOGDIR}`
zip -qr ${dir_name}.zip ${dir_name}
popd