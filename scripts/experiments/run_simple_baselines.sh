#!/bin/bash

CURRENT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SET_GLOBALVARS_PATH=${CURRENT_SCRIPT_DIR}/../setup/set_globalvars.sh

# Load the global vars
source ${SET_GLOBALVARS_PATH}

echo "Updating required git repositories."
${NAS_DMRL_PATH}/scripts/setup/setup_repositories.sh

echo "Running Default A2C"
${NAS_DMRL_PATH}/scripts/meta-rl/run_a2c.sh

echo "Running Meta A2C experiment"
${NAS_DMRL_PATH}/scripts/meta-rl/run_meta_a2c.sh