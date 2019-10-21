#!/bin/bash
set -e

################################################################################
## Script to convert the specified dataset to TFRecords using the             ##
## meta-dataset's code.                                                       ##
##                                                                            ##
## Assumptions:                                                               ##
##  - The desired dataset has already been downloaded. See:                   ##
##    scripts/meta-dataset/download_datasets.sh                               ##
##  - The meta-dataset's code is in ${WORKSPACE}/git_storage/meta-dataset     ##
##                                                                            ##
## The convertion happens in the directory:                                   ##
##   ${WORKSPACE}/metadataset_storage/records                                 ##
##                                                                            ##
## All the env. vars and options from scripts/setup/set_globalvars.sh and     ##
## scripts/setup/setup_condaenv.sh apply.                                     ##
##                                                                            ##
## Usage:                                                                     ##
##   convert_dataset.sh -d DATASET_NAME                                       ##
##                                                                            ##
## Supported datasets are: aircraft, cu_birds, dtd, fungi, imagenet,          ##
## omniglot, quickdraw, traffic_sign, vgg_flower.                             ##
################################################################################


CURRENT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SET_GLOBALVARS_PATH=${CURRENT_SCRIPT_DIR}/../setup/set_globalvars.sh

# Load the global vars
source ${SET_GLOBALVARS_PATH}

################################################################################
############ DEFINE THE GLOBAL VARIABLES AND ENVIRONMENT VARIABLES  ############
################################################################################

# Env vars required by meta-dataset's code
DATASRC_DIR=${WORKSPACE}/metadataset_storage
SPLITS_DIR=${DATASRC_DIR}/splits
RECORDS_DIR=${DATASRC_DIR}/records

################################################################################
####################### CREATE THE REQUIRED DIRECTORIES  #######################
################################################################################

if [ ! -d ${WORKSPACE} ]; then
    echo "Creating directory ${WORKSPACE}"
    mkdir ${WORKSPACE}
fi

if [ ! -d ${DATASRC_DIR} ]; then
    echo "Creating directory ${DATASRC_DIR}"
    mkdir -p ${DATASRC_DIR}
fi

if [ ! -d ${SPLITS_DIR} ]; then
    echo "Creating directory ${SPLITS_DIR}"
    mkdir -p ${SPLITS_DIR}
fi

if [ ! -d ${RECORDS_DIR} ]; then
    echo "Creating directory ${RECORDS_DIR}"
    mkdir -p ${RECORDS_DIR}
fi

################################################################################
####################### EXPORT THE ENVIRONMENT VARIABLES #######################
################################################################################

export DATASRC=${DATASRC_DIR}
export SPLITS=${SPLITS_DIR}
export RECORDS=${RECORDS_DIR}

################################################################################
######################### GET THE SCRIPT'S PATH & DIR  #########################
################################################################################

SCRIPT_PATH=`realpath $0`
SCRIPT_DIR=`dirname ${SCRIPT_PATH}`

################################################################################
############################### HELPER FUNCTIONS ###############################
################################################################################

usage() {
    echo "Usage:"
    echo ""
    echo "     convert_dataset.sh -d DATASET_NAME"
    echo ""
    echo "Supported datasets are: aircraft, cu_birds, dtd, fungi, imagenet, omniglot, quickdraw, traffic_sign, vgg_flower"
}

run_convertion() {
    pushd ${GIT_STORAGE}/meta-dataset
    echo "Executing conversion of dataset '${1}'"
    time python -m meta_dataset.dataset_conversion.convert_datasets_to_records \
        --dataset=${1} \
        --${1}_data_root=$DATASRC/${2} \
        --splits_root=$SPLITS \
        --records_root=$RECORDS
    popd
}

################################################################################
###################### READ THE REQUESTED DATASET IN ARGS ######################
################################################################################

while getopts ":d:" opt; do
  case ${opt} in
    d )
        DATASET_NAME=$OPTARG
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

if [ -z ${DATASET_NAME} ]; then
    usage
    exit 1
fi

################################################################################
########################## DOWNLOAD VIA THE SUBSCRIPT ##########################
################################################################################

case ${DATASET_NAME} in
    omniglot)
        run_convertion ${DATASET_NAME} ${DATASET_NAME}
        ;;
    imagenet)
        exit 0
        ;;
    aircraft)
        run_convertion ${DATASET_NAME} fgvc-aircraft-2013b
        ;;
    cu_birds)
        run_convertion ${DATASET_NAME} CUB_200_2011
        ;;
    dtd)
        run_convertion ${DATASET_NAME} ${DATASET_NAME}
        ;;
    quickdraw)
        run_convertion ${DATASET_NAME} ${DATASET_NAME}
        ;;
    fungi)
        run_convertion ${DATASET_NAME} ${DATASET_NAME}
        ;;
    vgg_flower)
        run_convertion ${DATASET_NAME} ${DATASET_NAME}
        ;;
    traffic_sign)
        run_convertion ${DATASET_NAME} GTSRB
        ;;
    mscoco)
        run_convertion ${DATASET_NAME} ${DATASET_NAME}
        ;;
    *)
        echo "Unkown dataset. Exiting with no errors..."
        exit 0
esac

exit 0