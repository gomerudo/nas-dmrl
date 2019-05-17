#!/bin/bash
set -e

################################################################################
############ DEFINE THE GLOBAL VARIABLES AND ENVIRONMENT VARIABLES  ############
################################################################################

TARGET_DIR=${DATASRC}/mscoco
IMGS_ZIP_NAME=train2017.zip
ANNOTATIONS_ZIP_NAME=annotations_trainval2017.zip

IMGS_ZIP_URL=http://images.cocodataset.org/zips/${IMGS_ZIP_NAME}
ANNOTATIONS_ZIP_URL=http://images.cocodataset.org/annotations/${ANNOTATIONS_ZIP_NAME}

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

if [ ! -f ${DATASRC}/${IMGS_ZIP_NAME} ]; then
    echo "Downloading ${IMGS_ZIP_URL} into ${DATASRC} ..."
    wget -P ${DATASRC}/ ${IMGS_ZIP_URL}
else
    echo "Skipping downloading. File ${DATASRC}/${IMGS_ZIP_NAME} already exists."
fi

if [ ! -f ${TARGET_DIR}/${ANNOTATIONS_ZIP_NAME} ]; then
    echo "Downloading ${ANNOTATIONS_ZIP_URL} into ${TARGET_DIR} ..."
    wget -P ${TARGET_DIR}/ ${ANNOTATIONS_ZIP_URL}
else
    echo "Skipping downloading. File ${DATASRC}/${ANNOTATIONS_ZIP_URL} already exists."
fi

################################################################################
############################## THE UNTAR PROCESS  ##############################
################################################################################

echo "Unzipping ${DATASRC}/${IMGS_ZIP_NAME} into ${TARGET_DIR}"
unzip -q ${DATASRC}/${IMGS_ZIP_NAME} -d ${TARGET_DIR}

echo "Unzipping ${DATASRC}/${ANNOTATIONS_ZIP_NAME} into ${TARGET_DIR}"
unzip -q ${DATASRC}/${ANNOTATIONS_ZIP_NAME} -d ${TARGET_DIR}
