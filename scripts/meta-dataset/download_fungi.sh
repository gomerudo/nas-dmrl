#!/bin/bash
set -e

################################################################################
############ DEFINE THE GLOBAL VARIABLES AND ENVIRONMENT VARIABLES  ############
################################################################################

TARGET_DIR=${DATASRC}/fungi
IMGS_TARGZ_NAME=fungi_train_val.tgz
ANNOTATIONS_TARGZ_NAME=train_val_annotations.tgz

IMGS_TARGZ_URL="https://data.deic.dk/public.php?service=files&t=2fd47962a38e2a70570f3be027cea57f&download"
ANNOTATIONS_TARGZ_URL="https://data.deic.dk/public.php?service=files&t=8dc110f312677d2b53003de983b3a26e&download"

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
    wget -O ${DATASRC}/${IMGS_TARGZ_NAME} ${IMGS_TARGZ_URL}
else
    echo "Skipping downloading. File ${DATASRC}/${IMGS_TARGZ_NAME} already exists."
fi

if [ ! -f ${DATASRC}/${ANNOTATIONS_TARGZ_NAME} ]; then
    echo "Downloading ${ANNOTATIONS_TARGZ_URL} into ${DATASRC} ..."
    wget -O ${DATASRC}/${ANNOTATIONS_TARGZ_NAME} ${ANNOTATIONS_TARGZ_URL}
else
    echo "Skipping downloading. File ${DATASRC}/${ANNOTATIONS_TARGZ_URL} already exists."
fi

################################################################################
############################## THE UNTAR PROCESS  ##############################
################################################################################

echo "Untaring ${DATASRC}/${IMGS_TARGZ_NAME} into ${TARGET_DIR}"
tar -C ${TARGET_DIR} -xzf ${DATASRC}/${IMGS_TARGZ_NAME}

echo "Untaring ${DATASRC}/${ANNOTATIONS_TARGZ_NAME} into ${TARGET_DIR}"
tar -C ${TARGET_DIR} -xzf ${DATASRC}/${ANNOTATIONS_TARGZ_NAME}
