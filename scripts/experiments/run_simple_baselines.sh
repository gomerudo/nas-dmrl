#!/bin/bash

CURRENT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SET_GLOBALVARS_PATH=${CURRENT_SCRIPT_DIR}/../setup/set_globalvars.sh

# Load the global vars
source ${SET_GLOBALVARS_PATH}

current_date=`date +%Y%m%d%H%M%S`
LOGDIR="${WORKSPACE}/results/opeanai-${current_date}"
export OPENAI_LOGDIR=${LOGDIR}

echo "Updating required git repositories."
${NAS_DMRL_PATH}/scripts/setup/setup_repositories.sh

# echo "Running Default A2C"
# ${NAS_DMRL_PATH}/scripts/meta-rl/run_a2c.sh

echo "Running Meta A2C experiment"
${NAS_DMRL_PATH}/scripts/meta-rl/run_meta_a2c.sh

pushd ${OPENAI_LOGDIR}
cd ..
dir_name=`basename ${OPENAI_LOGDIR}`
# Zip the results for easy export
zip -qr ${dir_name}.zip ${dir_name}
popd
