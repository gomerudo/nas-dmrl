#!/bin/bash

################################################################################
################################## FUNCTIONS  ##################################
################################################################################

usage() {
    echo "Usage:"
    echo ""
    echo "     run_evaluation.sh -c CONFIG_FILE"
    echo ""
}

parse_var() {
    echo `grep  -E "${1}\s*=\s*.+" ${INI_FILE} | sed 's/\([[:alpha:]]*[[:space:]]*=[[:space:]]*\)\(.*\)$/\2/'`
}

parse_ini_file() {
    GPU_MONITOR_SECONDS="$(parse_var GPUMonitorSec)"
    SLEEP_TIME_SECONDS="$(parse_var SleepTimeSec)"
    CONFIG_LOG_PATH="$(parse_var LogPath)"
}

################################################################################
################################ PARSE OPTIONS  ################################
################################################################################

while getopts ":c:m:r" opt; do
  case ${opt} in
    c )
        INI_FILE=$OPTARG
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

export WITH_CONDA=YES

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
############################## RUN THE EXPRIMENTS ##############################
################################################################################

# Let the nvidia monitoring running in the background for as long as the process runs
# echo "Running GPU monitoring tool in the background. It will log every ${GPU_MONITOR_SECONDS} seconds"
# nvidia-smi \
# --query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used \
# --format=csv \
# -l ${GPU_MONITOR_SECONDS} > ${EXPERIMENT_DIR}/smi-${START_TIMESTAMP}.csv 2>&1 &

if [ ! -d ${CONFIG_LOG_PATH} ]; then
    echo "Making workspace directory"
    mkdir ${CONFIG_LOG_PATH}
fi

command="time python ${CURRENT_SCRIPT_DIR}/policy_evaluation.py"
echo "Executing ${command}"

${command}
