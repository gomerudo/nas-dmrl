# Learning to reinforcement learn for Neural Architecture Search

MSc thesis project developed by Jorge Gomez Robles
at the Eindhoven University of Technology (TU/e), under the supervision of dr. ir. Joaquin Vanschoren.

You can check the paper [here](files/L2RL-NAS.pdf).


## Points of contact

 - Jorge Gomez Robles (j.gomezrb.dev@gmail.com)
 - Joaquin Vanschoren (j.vanschoren@gmail.com)

## Overview of the research project

The ultimate goal of Neural Architecture Search (NAS) is to come up with an algorithm that can design well-performing architectures for any dataset of interest. A promising approach to NAS is reinforcement learning (RL). 

One of the main limitations of RL on the NAS problem is the need to run the procedure from scratch for every dataset of interest. So far, most of the relevant results show how to apply standard RL algorithms on NAS for CIFAR, but little attention is paid to other datasets. Morever, RL tends to be an expensive procedure for NAS, making it unfeasible to replay it on new datasets.

An alternative is to explore meta-RL, which can learn a policy that can be transferred to previously unseen environments (i.e., datasets). In this work, we explore, for the first time, meta-RL for NAS. We study whether or not the transfer provides an advantage during traning and evaluation (i.e., when the policy is fixed).

The meta-RL algorithm that we use is inspired by the work of [Wang et al.](https://arxiv.org/abs/1611.05763) and [Duan et al.](https://arxiv.org/abs/1611.02779). Our NAS search space and performance estimation strategy are based on the [BlockQNN](https://arxiv.org/abs/1808.05584) methodology. The environments are associated to 5 datasets from the [meta-dataset](https://arxiv.org/abs/1903.03096): `omniglot`, `vgg_flower`, and `dtd` for training; `aircraft` and `cu_birds` for evaluation.

## Results

### Experiment 1: training the meta-RL agent to design chain-structured networks

### Experiment 2: evaluating the policy on previously unseen environments

### Experiment 3: training the meta-RL agent to design multi-branch structures


## How to run

### Requirements
This repository contains all scripts and configuration files required to run the experiments. To run 
them it is necessary to have installed the [Open AI Gym](https://gym.openai.com) and the 
[NasGym](https://github.com/gomerudo/nas-env). Also, the next repositories are required: [custom openai-baselines](https://github.com/gomerudo/openai-baselines/tree/experiments/baselines/meta_a2c), [nas-env](https://github.com/gomerudo/nas-env), and [custom meta-dataset](https://github.com/gomerudo/meta-dataset).

### Setup Instructions

We recommend to follow the next steps to avoid unnecessary changes in the scripts:

1. Clone all needed repositores into a directory `~/workspace/git_storage`.
2. Install mini-conda and setup an environment with name `nasdmrl` and Python 3.6.8
3. Run [scripts/setup/install_pypkgs.sh]()
4. Modify the paths in the nasenv.yml files under [configs/]()

## Run experiments

The experiments can be run with the [scripts/experiments/run_nas_dmrl.sh]() script.
