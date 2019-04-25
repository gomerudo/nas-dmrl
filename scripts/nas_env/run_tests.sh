#!/bin/bash

CURRENT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SET_GLOBALVARS_PATH=${CURRENT_SCRIPT_DIR}/../setup/set_globalvars.sh

# Load the global vars
source ${SET_GLOBALVARS_PATH}
# Load the conda script
source ${SETUP_CONDA_PATH}
# Source the environment
conda activate ${VENV_NAME}

pushd ${NASENV_PATH}

# Remove directories to ensure
rm -rf workspace/trainer_test
rm -rf workspace/trainer_test_earlystop

time python -m unittest -v \
test/test_default_db.py \
test/test_net_builder.py \
test/test_net_trainer.py \
test/test_default_nasenv.py
popd

conda deactivate