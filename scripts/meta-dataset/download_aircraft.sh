#!/bin/bash
set -e

################################################################################
############ DEFINE THE GLOBAL VARIABLES AND ENVIRONMENT VARIABLES  ############
################################################################################

TARGET_DIR=${DATASRC}/fgvc-aircraft-2013b
DATA_TARGZ_NAME=fgvc-aircraft-2013b.tar.gz
DATA_ZIP_URL=http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/${DATA_TARGZ_NAME}

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
    echo "Downloading ${DATA_ZIP_URL} into ${DATASRC} ..."
    wget -P ${DATASRC}/ ${DATA_ZIP_URL}
else
    echo "Skipping downloading. File ${DATASRC}/${DATA_TARGZ_NAME} already exists."
fi

################################################################################
############################## THE UNTAR PROCESS  ##############################
################################################################################

tar -C ${TARGET_DIR} -xf ${DATASRC}/${DATA_TARGZ_NAME}