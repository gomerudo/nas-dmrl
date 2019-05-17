#!/bin/bash
set -e

################################################################################
############ DEFINE THE GLOBAL VARIABLES AND ENVIRONMENT VARIABLES  ############
################################################################################

TARGET_DIR=${DATASRC}/vgg_flower
IMGS_TARGZ_NAME=102flowers.tgz
ANNOTATIONS_NAME=imagelabels.mat

IMGS_TARGZ_URL=http://www.robots.ox.ac.uk/~vgg/data/flowers/102/${IMGS_TARGZ_NAME}
ANNOTATIONS_URL=http://www.robots.ox.ac.uk/~vgg/data/flowers/102/${ANNOTATIONS_NAME}

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

if [ ! -f ${DATASRC}/${IMGS_TARGZ_NAME} ]; then
    echo "Downloading ${IMGS_TARGZ_URL} into ${DATASRC} ..."
    wget -P ${DATASRC}/ ${IMGS_TARGZ_URL}
else
    echo "Skipping downloading. File ${DATASRC}/${IMGS_TARGZ_NAME} already exists."
fi

if [ ! -f ${TARGET_DIR}/${ANNOTATIONS_NAME} ]; then
    echo "Downloading ${ANNOTATIONS_URL} into ${TARGET_DIR} ..."
    wget -P ${TARGET_DIR}/ ${ANNOTATIONS_URL}
else
    echo "Skipping downloading. File ${DATASRC}/${ANNOTATIONS_URL} already exists."
fi

################################################################################
############################## THE UNTAR PROCESS  ##############################
################################################################################

echo "Untaring ${DATASRC}/${IMGS_TARGZ_NAME} into ${TARGET_DIR}"
tar -C ${TARGET_DIR} -xzf ${DATASRC}/${IMGS_TARGZ_NAME}
