#!/bin/bash
set -e

################################################################################
############ DEFINE THE GLOBAL VARIABLES AND ENVIRONMENT VARIABLES  ############
################################################################################

DATASET_DIRNAME=ILSVRC2012_img_train
DATASET_TAR_NAME=${DATASET_DIRNAME}.tar
DOWNLOAD_URL=http://www.image-net.org/challenges/LSVRC/2012/nnoupb/${DATASET_TAR_NAME}

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

tar -C ${DATASRC} -xf ${DATASRC}/${DATASET_TAR_NAME}