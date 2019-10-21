#!/bin/bash
set -e

################################################################################
############ DEFINE THE GLOBAL VARIABLES AND ENVIRONMENT VARIABLES  ############
################################################################################

TARGET_DIR=${DATASRC}/dtd
DATA_TARGZ_NAME=dtd-r1.0.1.tar.gz
DATA_TARGZ_URL=https://www.robots.ox.ac.uk/~vgg/data/dtd/download/${DATA_TARGZ_NAME}

################################################################################
########################## MAKE REQUIRED DIRECTORIES  ##########################
################################################################################

if [ ! -d ${TARGET_DIR} ]; then
    echo "Creating directory ${TARGET_DIR}"
    mkdir -p ${TARGET_DIR}
fi

################################################################################
############################# THE DOWNLOAD PROCESS #############################
################################################################################

if [ ! -f ${DATASRC}/${DATA_TARGZ_NAME} ]; then
    echo "Downloading ${DATA_TARGZ_URL} into ${DATASRC} ..."
    wget -P ${DATASRC}/ ${DATA_TARGZ_URL}
else
    echo "Skipping downloading. File ${DATASRC}/${DATA_TARGZ_NAME} already exists."
fi

################################################################################
############################## THE UNTAR PROCESS  ##############################
################################################################################

echo "Untaring the dataset ..."
tar -C ${DATASRC} -xzf ${DATASRC}/${DATA_TARGZ_NAME}