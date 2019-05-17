#!/bin/bash
set -e

################################################################################
############ DEFINE THE GLOBAL VARIABLES AND ENVIRONMENT VARIABLES  ############
################################################################################

# Global vars
WORKSPACE=${HOME}/workspace

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
    echo "     download_dataset.sh -d DATASET_NAME"
    echo ""
    echo "Supported datasets are: imagenet, omniglot"
}

run_custom_download() {
    SUBSSCRIPT=${SCRIPT_DIR}/download_${1}.sh
    echo "Calling subscript: ${SUBSSCRIPT}"
    sh ${SUBSSCRIPT}
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
        run_custom_download "onmiglot"
        ;;
    imagenet)
        run_custom_download "imagenet"
        ;;
    aircraft)
        run_custom_download "aircraft"
        ;;
    cu_birds)
        run_custom_download "cu_birds"
        ;;
    dtd)
        run_custom_download "dtd"
        ;;
    quickdraw)
        run_custom_download "quickdraw"
        ;;
    fungi)
        run_custom_download "fungi"
        ;;
    vgg_flower)
        run_custom_download "vgg_flower"
        ;;
    traffic_sign)
        run_custom_download "traffic_sign"
        ;;
    mscoco)
        run_custom_download "mscoco"
        ;;
    *)
        echo "Unkown dataset. Exiting with no errors..."
        exit 0
esac

exit 0