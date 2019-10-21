#!/bin/bash
set -e

################################################################################
############ DEFINE THE GLOBAL VARIABLES AND ENVIRONMENT VARIABLES  ############
################################################################################

DATA_TARGZ_NAME=fgvc-aircraft-2013b.tar.gz
DATA_ZIP_URL=http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/${DATA_TARGZ_NAME}

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

echo "Untaring the dataset ..."
tar -C ${DATASRC} -xzf ${DATASRC}/${DATA_TARGZ_NAME}