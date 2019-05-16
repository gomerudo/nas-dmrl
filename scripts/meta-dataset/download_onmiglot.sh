#!/bin/bash
set -e

################################################################################
############ DEFINE THE GLOBAL VARIABLES AND ENVIRONMENT VARIABLES  ############
################################################################################

TARGET_DIR=${DATASRC}/omniglot
BACKGROUND_ZIP_NAME=images_background.zip
EVALUATION_ZIP_NAME=images_evaluation.zip

BACKGROUND_ZIP_URL=https://github.com/brendenlake/omniglot/raw/master/python/${BACKGROUND_ZIP_NAME}
EVALUATION_ZIP_URL=https://github.com/brendenlake/omniglot/raw/master/python/${EVALUATION_ZIP_NAME}

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

if [ ! -f ${DATASRC}/${BACKGROUND_ZIP_NAME} ]; then
    echo "Downloading ${BACKGROUND_ZIP_URL} into ${DATASRC} ..."
    wget -P ${DATASRC}/ ${BACKGROUND_ZIP_URL}
else
    echo "Skipping downloading. File ${DATASRC}/${BACKGROUND_ZIP_NAME} already exists."
fi

if [ ! -f ${DATASRC}/${EVALUATION_ZIP_NAME} ]; then
    echo "Downloading ${EVALUATION_ZIP_URL} into ${DATASRC} ..."
    wget -P ${DATASRC}/ ${EVALUATION_ZIP_URL}
else
    echo "Skipping downloading. File ${DATASRC}/${EVALUATION_ZIP_URL} already exists."
fi

################################################################################
############################## THE UNZIP PROCESS  ##############################
################################################################################

echo "Unzipping ${DATASRC}/${BACKGROUND_ZIP_NAME} into ${${TARGET_DIR}}"
unzip -q ${DATASRC}/${BACKGROUND_ZIP_NAME} -d ${TARGET_DIR}

echo "Unzipping ${DATASRC}/${EVALUATION_ZIP_NAME} into ${${TARGET_DIR}}"
unzip -q ${DATASRC}/${EVALUATION_ZIP_NAME} -d ${TARGET_DIR}