# Scripts for meta-dataset

## Overview

The meta-dataset ([Triantafillou et al.](https://arxiv.org/abs/1903.03096)) is a collection of datasets for few-shot classification. The datasets are available online but in formats that are not computationally efficient. Hence the meta-dataset authors have released code to export them to TFRecords files. 

In this folder, we provide scripts to download and convert the datasets. Since originally the meta-dataset code was engineered to work with Python 2 only, the scripts available here rely on a [custom meta-dataset repository forked](https://github.com/gomerudo/meta-dataset) from the original one. Below you will find some instructions on how to use the scripts.

**Note:** as of the latest commit of this repository, the original meta-dataset now works with Python 3. We recommend to check it before using our code.

## TU/e staff notes

For TU/e staff with access to the internal DM cluster, the download can be skipped since the downloaded datasets and their TFRecords version are available at `/home/TUE/20175601/workspace/metadataset_storage`. You should simply copy all the contents into your own `${HOME}/workspace/metadataset_storage` and setup your environment as indicated in the next section.

## Environment setup

The scripts available here rely on a customized setup. To replicate it and avoid errors or unnecessary modifications, follow the next steps

1. Create the workspace: `mkdir ${HOME}/workspace`
2. Export the environment variable: `WORKSPACE=${HOME}/workspace`
3. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) into `${WORKSPACE}`, so that the miniconda path is `${WORKSPACE}/miniconda3`
4. Create the virtual environment `nasdmrl` with Python 3.6.8: `conda create -n nasdmrl python=3.6.8`
5. Create the git storage as follows: `mkdir ${WORKSPACE}/git_storage` and associate it to an environment variable named `${GIT_STORAGE}`.
6. Clone our version of the [meta-dataset](https://github.com/gomerudo/meta-dataset) into `${GIT_STORAGE}` and make sure to be in the branch `develop`
7. Clone this repository (nas-dmrl) and make sure to be in the branch `master`.
8. Activate the `nasdmrl` environment: `conda activate nasdmrl`
9. Install the next packages via `conda install`: `cloudpickle`, `Pillow=5.4.1`, `scikit-learn`, `pandas`, `tensorflow`.
10. Install the next packages via `pip install`: `opencv-python`, `gin-config`.

**Note:** Steps 9 and 10 can be skipped if you rely on our `scripts/setup/install_pypkgs.sh` script.

## Download the datasets

To download the datasets into `${WORKSPACE}/metadataset_storage` follow the next steps:

```
cd ${GIT_STORAGE}/nas-dmrl
scripts/meta-dataset/download_datasets.sh -d [DATASET_ID]
```

- Allowed values for `DATASET_ID` in this script: `aircraft`, `cu_birds`, `dtd`, `fungi`, `imagenet`, `mscoco`, `omniglot`, `quickdraw`, `traffic_sign`, `vgg_flower`.

## Convert the datasets

To convert the original datasets to TFRecords files run the next commands:

```
cd ${GIT_STORAGE}/nas-dmrl
scripts/meta-dataset/convert_dataset.sh -d [DATASET_ID]
```

- Allowed values for `DATASET_ID` in this script: `aircraft`, `cu_birds`, `dtd`, `fungi`, `imagenet`, `omniglot`, `quickdraw`, `traffic_sign`, `vgg_flower`.

**Note**: `mscoco` is not available for convertion because of an error in the original meta-dataset code. We recommend to check the [original meta-dataset repository](https://github.com/google-research/meta-dataset) to check if a fix has already been introduced.

## Explore the contents of the TFRecords files

It would be desirable to see what are the available classes and the number of observations per dataset, to do so, run the following command:

```
cd ${GIT_STORAGE}/nas-dmrl
scripts/meta-dataset/run_explorer.sh -d DATASET_ID

```

- Allowed values for `DATASET_ID` in this script: `aircraft`, `cu_birds`, `dtd`, `fungi`, `imagenet`, `omniglot`, `quickdraw`, `traffic_sign`, `vgg_flower`.

## Export TFRecords to CSV file

To convert the TFRecords to a CSV format, run the next commands:

```
cd ${GIT_STORAGE}/nas-dmrl
scripts/meta-dataset/export_dataset.sh -d DATASET_ID -t TARGET_FILE [-s IMAGE_SIZE]
```

- Allowed values for `DATASET_ID` in this script: `aircraft`, `cu_birds`, `dtd`, `fungi`, `imagenet`, `omniglot`, `quickdraw`, `traffic_sign`, `vgg_flower`.
- `TARGET_FILE` is the path of the destination CSV file.
- `IMAGE_SIZE` is an integer value specifying the dimenstion of the desired images after resizing with the meta-dataset's original resizing procedure. Default is `84`, meaning images are resized to `84x84`.


