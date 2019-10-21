#!/bin/bash
set -e

################################################################################
## Export a meta-dataset's dataset to a CSV file.                             ##
##                                                                            ##
## Assumes that the TFRecords exist in the directory                          ##
## ${WORKSPACE}/metadataset_storage/records/DATASET_ID                        ##
##                                                                            ##
## All the env. vars and options from scripts/setup/set_globalvars.sh and     ##
## scripts/setup/setup_condaenv.sh apply.                                     ##
##                                                                            ##
## Usage:                                                                     ##
##   export_dataset.sh -d DATASET_ID -t TARGET_FILE [-s IMAGE_SIZE]           ##
##                                                                            ##
## - Default IMAGE_SIZE: 84                                                   ##
## - Supported datasets are: aircraft, cu_birds, dtd, fungi, imagenet,        ##
##   omniglot, quickdraw, traffic_sign, vgg_flower.                           ##
################################################################################

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
    echo "     export_dataset.sh -d DATASET_ID -t TARGET_FILE [-s IMAGE_SIZE]"
    echo ""
    echo "Supported IDs are: aircraft, cu_birds, dtd, fungi, imagenet, omniglot, quickdraw, traffic_sign, vgg_flower"
}

while getopts ":d:s:t:" opt; do
  case ${opt} in
    d )
        DATASET_ID=$OPTARG
        ;;
    s )
        IMG_SIZE=$OPTARG
        ;;
    t )
        TARGET_FILE=$OPTARG
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

if [ -z ${TFRECORDS_DIR} ]; then
    TFRECORDS_DIR=${WORKSPACE}/metadataset_storage/records/${DATASET_ID}
fi

if [ ! -d ${TFRECORDS_DIR} ]; then
    echo "Directory ${TFRECORDS_DIR} does not exist"
    exit 1
fi

if [ -z ${IMG_SIZE} ]; then
    IMG_SIZE=84
fi

if [ -z ${TARGET_FILE} ]; then
    echo "Target file has to be provided"
    usage
fi

echo "Running conversion via python script"
python ${GIT_STORAGE}/nas-dmrl_par/scripts/meta-dataset/tfrecords_exporter.py \
    --src_dir=${TFRECORDS_DIR} \
    --imgsize=${IMG_SIZE} \
    --target_file=${TARGET_FILE}
