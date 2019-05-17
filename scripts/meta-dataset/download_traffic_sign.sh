#!/bin/bash
set -e

################################################################################
############ DEFINE THE GLOBAL VARIABLES AND ENVIRONMENT VARIABLES  ############
################################################################################

TARGET_DIR=${DATASRC}/GTSRB
DATA_ZIP_NAME=GTSRB_Final_Training_Images.zip
DATA_ZIP_URL=http://benchmark.ini.rub.de/Dataset/${DATA_ZIP_NAME}

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

if [ ! -f ${DATASRC}/${DATA_ZIP_NAME} ]; then
    echo "Downloading ${DATA_ZIP_URL} into ${DATASRC} ..."
    wget -P ${DATASRC}/ ${DATA_ZIP_URL}
else
    echo "Skipping downloading. File ${DATASRC}/${DATA_ZIP_NAME} already exists."
fi

################################################################################
############################## THE UNZIP PROCESS  ##############################
################################################################################

echo "Unzipping the dataset ..."
unzip -q ${DATASRC}/${DATA_ZIP_NAME} -d ${TARGET_DIR}
