#!/bin/bash

################################################################################
################################## FUNCTIONS  ##################################
################################################################################

usage() {
    echo "Usage:"
    echo ""
    echo "     run_nas_random.sh -c CONFIG_FILE [-r]"
    echo ""
}

parse_var() {
    echo `grep  -E "${1}\s*=\s*.+" ${INI_FILE} | sed 's/\([[:alpha:]]*[[:space:]]*=[[:space:]]*\)\(.*\)$/\2/'`
}

parse_ini_file() {
    N_TIMESTEPS="$(parse_var NumTimesteps)"
    RANDOM_SEED="$(parse_var RandomSeedGlobal)"
    GPU_MONITOR_SECONDS="$(parse_var GPUMonitorSec)"
    CONFIG_LOG_PATH="$(parse_var LogPath)"
}

################################################################################
################################ PARSE OPTIONS  ################################
################################################################################

while getopts ":c:r" opt; do
  case ${opt} in
    c )
        INI_FILE=$OPTARG
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
########################## MAKE EXPERIMENTS DIRECTORY ##########################
################################################################################
EXPERIMENT_DIR="/scratch/nodespecific/int1/jgomes/results/experiment-${START_TIMESTAMP}"
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

# Let the nvidia monitoring running in the background for as long as the process runs
echo "Running GPU monitoring tool in the background. It will log every ${GPU_MONITOR_SECONDS} seconds"
nvidia-smi \
--query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used \
--format=csv \
-l ${GPU_MONITOR_SECONDS} > ${EXPERIMENT_DIR}/smi-${START_TIMESTAMP}.csv 2>&1 &

pushd ${OPENAI_BASELINES_PATH}

if [ ! -d ${CONFIG_LOG_PATH} ]; then
    echo "Making workspace directory"
    mkdir ${CONFIG_LOG_PATH}
fi

export LIMITED_STORAGE=YES

command="time python ${CURRENT_SCRIPT_DIR}/random_search.py \
--log_dir=${EXPERIMENT_DIR} \
--ntimesteps=${N_TIMESTEPS} \
--random_seed=${RANDOM_SEED}"

echo "Command to execute is: ${command}"
${command}

if [ ! -z ${REMOVE} ]; then
    rm -rf ${CONFIG_LOG_PATH}/trainer*
fi

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
