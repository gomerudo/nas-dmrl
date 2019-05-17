#!/bin/bash
set -e

################################################################################
############ DEFINE THE GLOBAL VARIABLES AND ENVIRONMENT VARIABLES  ############
################################################################################

TARGET_DIR=${DATASRC}/quickdraw
SCRIPT_PATH=`realpath $0`
SCRIPT_DIR=`dirname ${SCRIPT_PATH}`

FILES_LIST_TXT=${SCRIPT_DIR}/quickdraw_files.txt
ROOT_DOWNLOAD_URL=https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap
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

while read LINE; do
    full_url=${ROOT_DOWNLOAD_URL}/${LINE}
    wget -P ${TARGET_DIR}/ ${full_url}
done < ${FILES_LIST_TXT}
