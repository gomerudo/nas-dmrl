#!/bin/bash
set -e

CURRENT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SET_GLOBALVARS_PATH=${CURRENT_SCRIPT_DIR}/../setup/set_globalvars.sh

# Load the global vars
source ${SET_GLOBALVARS_PATH}
# Load the conda script
source ${SETUP_CONDA_PATH}
# Source the environment
conda activate ${VENV_NAME}

################################################################################
########################### READ THE REQUESTED ARGS  ###########################
################################################################################

usage() {
    echo "Usage:"
    echo ""
    echo "     run_explorer.sh -d DATASET_ID"
    echo ""
    echo "Supported IDs are: aircraft, cu_birds, dtd, fungi, imagenet, omniglot, quickdraw, traffic_sign, vgg_flower"
}

while getopts ":d:" opt; do
  case ${opt} in
    d )
        DATASET_ID=$OPTARG
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

if [ -z ${DATASET_ID} ]; then
    usage
    exit 1
fi

TFRECORDS_DIR=${WORKSPACE}/metadataset_storage/records/${DATASET_ID}
if [ ! -d ${TFRECORDS_DIR} ]; then
    echo "No directory ${TFRECORDS_DIR}"
    exit 1
fi

python ${GIT_STORAGE}/nas-dmrl/scripts/meta-dataset/tfrecords_explorer.py \
    --path=${TFRECORDS_DIR}