#!/bin/bash
set -e

################################################################################
############ DEFINE THE GLOBAL VARIABLES AND ENVIRONMENT VARIABLES  ############
################################################################################

# Global vars
WORKSPACE=${HOME}/workspace

# Env vars required by meta-dataset's code
DATASRC_DIR=${WORKSPACE}/metadataset_storage
SPLITS_DIR=${DATASRC}/splits
RECORDS_DIR=${DATASRC}/records

# Define URL's
http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar

################################################################################
####################### CREATE THE REQUIRED DIRECTORIES  #######################
################################################################################

if [ ! -d ${WORKSPACE} ]; then
    mkdir ${WORKSPACE}
fi

if [ ! -d ${DATASRC_DIR} ]; then
    mkdir -p ${DATASRC_DIR}
fi

if [ ! -d ${SPLITS_DIR} ]; then
    mkdir -p ${SPLITS_DIR}
fi

if [ ! -d ${RECORDS_DIR} ]; then
    mkdir -p ${RECORDS_DIR}
fi

################################################################################
####################### EXPORT THE ENVIRONMENT VARIABLES #######################
################################################################################

export DATASRC=${DATASRC_DIR}
export SPLITS=${SPLITS_DIR}
export RECORDS=${RECORDS_DIR}



