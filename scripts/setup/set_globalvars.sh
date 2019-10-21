#!/bin/bash

################################################################################
## Set the global variables required by the rest of the scripts in this repo. ##
##                                                                            ##
## All scripts assume the directory structure intrinsically specified here.   ##
################################################################################

# The most general global variables
WORKSPACE=${HOME}/workspace
GIT_STORAGE=${WORKSPACE}/git_storage

# The global variables for the git repositories
OPENAI_BASELINES_PATH=${GIT_STORAGE}/openai-baselines
NASENV_PATH=${GIT_STORAGE}/nas-env
NAS_DMRL_PATH=${GIT_STORAGE}/nas-dmrl

# Conda
SETUP_SCRIPTS_DIR=${NAS_DMRL_PATH}/scripts/setup
SETUP_CONDA_PATH=${SETUP_SCRIPTS_DIR}/setup_condaenv.sh