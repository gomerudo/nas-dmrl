#!/bin/bash

################################################################################
################################## FUNCTIONS  ##################################
################################################################################

usage() {
    echo "Usage:"
    echo ""
    echo "     run_nas_dmrl.sh -c CONFIG_FILE [-m PREWARM_MODEL] [-r]"
    echo ""
}

parse_var() {
    echo `grep  -E "${1}\s*=\s*.+" ${INI_FILE} | sed 's/\([[:alpha:]]*[[:space:]]*=[[:space:]]*\)\(.*\)$/\2/'`
}

parse_ini_file() {
    RL_ALGORITHM="$(parse_var Algorithm)"
    RL_ENVIRONMENT="$(parse_var Environment)"
    RL_NETWORK="$(parse_var Network)"
    FINAL_MODEL_NAME="$(parse_var FinalModelName)"
    N_TASKS="$(parse_var NTasks)"
    N_STEPS="$(parse_var NSteps)"
    N_TIMESTEPS="$(parse_var NumTimesteps)"
    N_TRIALS="$(parse_var NTrials)"
    GPU_MONITOR_SECONDS="$(parse_var GPUMonitorSec)"
    SLEEP_TIME_SECONDS="$(parse_var SleepTimeSec)"
    CONFIG_LOG_PATH="$(parse_var LogPath)"
    LOG_INTERVAL="$(parse_var LogInterval)"
}

################################################################################
################################ PARSE OPTIONS  ################################
################################################################################

while getopts ":c:m:r" opt; do
  case ${opt} in
    c )
        INI_FILE=$OPTARG
        ;;
    m )
        WARMUP_MODEL=$OPTARG
        ;;
    r )
        REMOVE=YES
        ;;
    \? )
        usage
        exit 1
        ;;
    : )
        usage
        exit 1
        ;;
  esac
done

################################################################################
########################### VALIDATE MANDATORY ARGS  ###########################
################################################################################

if [ -z ${INI_FILE} ]; then
    usage
    exit 1
fi

################################################################################
################################ LOG ARGUMENTS  ################################
################################################################################

echo "Experiment will use the configuration in file ${INI_FILE}"
export NAS_DMRL_CONFIG_FILE=${INI_FILE}

if [ ! -z ${WARMUP_MODEL} ]; then
    echo "Experiment will use model ${WARMUP_MODEL} as a pre-warm model."
fi

################################################################################
################################ GET MASTER PID  ################################
################################################################################

MASTER_PID=$$
echo "Master process PID: ${MASTER_PID}"

################################################################################
############################### PARSE CONFIG.INI ###############################
################################################################################

parse_ini_file

################################################################################
############################# SET GLOBAL VARIABLES #############################
################################################################################

# Obtain the script's current directory
CURRENT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Set the global environment variables
SET_GLOBALVARS_PATH=${CURRENT_SCRIPT_DIR}/../setup/set_globalvars.sh
source ${SET_GLOBALVARS_PATH}

# Load the conda script
source ${SETUP_CONDA_PATH}
# Source the environment
echo "Activating conda environment ${VENV_NAME}"
conda activate ${VENV_NAME}

# The start timestamp
START_TIMESTAMP=`date +%Y%m%d%H%M%S`

################################################################################
########################## MAKE EXPERIMENTS DIRECTORY ##########################
################################################################################
EXPERIMENT_DIR="${WORKSPACE}/results/experiment-${START_TIMESTAMP}"
echo "Directory for the experiment is ${EXPERIMENT_DIR}"

if [ ! -d ${EXPERIMENT_DIR} ]; then
    echo "Making directory for the experiment"
    mkdir -p ${EXPERIMENT_DIR}
fi

################################################################################
############################## RUN THE EXPRIMENTS ##############################
################################################################################

# Copy the configuration file so that we can know what we config we used
cp ${INI_FILE} ${EXPERIMENT_DIR}/.

echo "A total of ${N_TRIALS} trials will be executed"

# Let the nvidia monitoring running in the background for as long as the process runs
#echo "Running GPU monitoring tool in the background. It will log every ${GPU_MONITOR_SECONDS} seconds"
#nvidia-smi \
#--query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used \
#--format=csv \
#-l ${GPU_MONITOR_SECONDS} > ${EXPERIMENT_DIR}/smi-${START_TIMESTAMP}.csv 2>&1 &

pushd ${OPENAI_BASELINES_PATH}

if [ ! -d ${CONFIG_LOG_PATH} ]; then
    echo "Making workspace directory"
    mkdir ${CONFIG_LOG_PATH}
fi

for trial in $(seq 1  1 ${N_TRIALS}); do
    echo "Starting trial ${trial}"

    # We sleep for few seconds to let the memory release and the environment stabilize
    # Each trial will have its one timestamp to isolate the subexperiments
    TRIAL_TIMESTAMP=`date +%Y%m%d%H%M%S`

    # We obtain the last OPENAI_LOGDIR in the environmet, to use it as the
    # warmup model.
    LAST_OPENAI_LOGDIR="${OPENAI_LOGDIR}"

    # Export the OPENAI_LOGDIR as needed by openai-baselines project
    LOGDIR="${EXPERIMENT_DIR}/openai-${TRIAL_TIMESTAMP}"
    export OPENAI_LOGDIR=${LOGDIR}

    # Define the place where we will store the resulting model
    SAVE_DIR=${OPENAI_LOGDIR}/models

    if [ ! -d ${SAVE_DIR} ]; then
        mkdir -p ${SAVE_DIR}
    fi

# python -m baselines.run --alg=deepq --env=PongNoFrameskip-v4 --network=mlp --num_timesteps=1e5
    command="time python -m baselines.run \
--alg=${RL_ALGORITHM} \
--env=${RL_ENVIRONMENT} \
--network=${RL_NETWORK} \
--save_path=${SAVE_DIR}/${FINAL_MODEL_NAME} \
--buffer_size=20 \
--lr=0.01 \
--exploration_fraction=0.5 \
--exploration_final_eps=0.1 \
--learning_starts=${LOG_INTERVAL} \
--checkpoint_freq=${LOG_INTERVAL} \
--num_timesteps=${N_TIMESTEPS}"

    if [ ${trial} -eq 1 ]; then
        if [ ! -z ${WARMUP_MODEL} ]; then
            command="${command} --load_path=${WARMUP_MODEL}"
        fi
    else
        if [ ! -z ${LAST_OPENAI_LOGDIR} ]; then
            command="${command} --load_path=${LAST_OPENAI_LOGDIR}/models/${FINAL_MODEL_NAME}"
        fi
    fi

    echo "Command to execute is: ${command}"
    ${command}
    
    if [ ! -z ${REMOVE} ]; then
        rm -rf ${CONFIG_LOG_PATH}/trainer*
    fi

    # Always wait after the command has been executed
    sleep ${SLEEP_TIME_SECONDS}
done
popd

# Copy the db of experiments and the actions_info
cp ${CONFIG_LOG_PATH}/db_experiments.csv ${EXPERIMENT_DIR}/.
cp ${CONFIG_LOG_PATH}/actions_info.csv ${EXPERIMENT_DIR}/.

# Go to the Experiments' parent directory and zip the results
pushd ${EXPERIMENT_DIR}/..
dir_name=`basename ${EXPERIMENT_DIR}`
echo "Zipping the experiments ..."
zip -qr ${dir_name}.zip ${dir_name}
popd

