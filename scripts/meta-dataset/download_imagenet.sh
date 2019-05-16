#!/bin/bash
set -e

################################################################################
############ DEFINE THE GLOBAL VARIABLES AND ENVIRONMENT VARIABLES  ############
################################################################################

DATASET_DIRNAME=ILSVRC2012_img_train
DATASET_TAR_NAME=${DATASET_DIRNAME}.tar
DOWNLOAD_URL=http://www.image-net.org/challenges/LSVRC/2012/nnoupb/${DATASET_TAR_NAME}

if [ ! -f ]; then
    wget -P ${DATASRC_DIR}/ ${DOWNLOAD_URL}
else
    echo "Skipping downloading. File ${DATASRC_DIR}/${DATASET_TAR_NAME} already exists."
fi

tar -C ${DATASRC_DIR} -xf ${DATASRC_DIR}/${DATASET_TAR_NAME}