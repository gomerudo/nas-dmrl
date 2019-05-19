#!/bin/bash
set -e

################################################################################
############ DEFINE THE GLOBAL VARIABLES AND ENVIRONMENT VARIABLES  ############
################################################################################

TARGET_DIR=${DATASRC}/ILSVRC2012_img_train
DATASET_TAR_NAME=ILSVRC2012_img_train.tar
DOWNLOAD_URL=http://www.image-net.org/challenges/LSVRC/2012/nnoupb/${DATASET_TAR_NAME}


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

if [ ! -f ${DATASRC}/${DATASET_TAR_NAME} ]; then
    echo "Downloading ${DOWNLOAD_URL} into ${DATASRC} ..."
    wget -P ${DATASRC}/ ${DOWNLOAD_URL}
else
    echo "Skipping downloading. File ${DATASRC}/${DATASET_TAR_NAME} already exists."
fi

################################################################################
############################## THE UNTAR PROCESS  ##############################
################################################################################

echo "Untaring ${DATASRC}/${DATASET_TAR_NAME} into ${TARGET_DIR}"
tar -C ${TARGET_DIR} -xf ${DATASRC}/${DATASET_TAR_NAME}