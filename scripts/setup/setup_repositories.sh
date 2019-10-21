#!/bin/bash

CURRENT_SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SET_GLOBALVARS_PATH=${CURRENT_SCRIPT_DIR}/set_globalvars.sh

# Load the global vars
source ${SET_GLOBALVARS_PATH}

# OpenAI baselines repo
METARL_REPO_URL=https://github.com/gomerudo/openai-baselines.git
METARL_REPO_BRANCH=experiments
METARL_REPO_NAME=openai-baselines

# NasGym repo
NASENV_REPO_URL=https://github.com/gomerudo/nas-env.git
NASENV_REPO_BRANCH=develop
NASENV_REPO_NAME=nas-env

if [ ! -d ${GIT_STORAGE} ]; then
    echo "Creating git_storage"
    # mkdir ${GIT_STORAGE}
fi

pushd ${GIT_STORAGE}

# 1. Clone the openai-baselines repository
if [ -d ${GIT_STORAGE}/${METARL_REPO_NAME} ]; then
    pushd ${GIT_STORAGE}/${METARL_REPO_NAME}
    git fetch
    git pull origin ${METARL_REPO_BRANCH}
    popd
else
    git clone -b ${METARL_REPO_BRANCH} ${METARL_REPO_URL}
fi

# 2. Clone the nas-env repository
if [ -d ${GIT_STORAGE}/${NASENV_REPO_NAME} ]; then
    git fetch
    pushd ${GIT_STORAGE}/${NASENV_REPO_NAME}
    git pull origin ${NASENV_REPO_BRANCH}
    popd
else
    git clone -b ${NASENV_REPO_BRANCH} ${NASENV_REPO_URL}
fi

popd