# Learning to reinforcement learn for Neural Architecture Search

Thesis project for Neural Architecture Search using Deep meta-reinforcement
learning. The project is currently **under development**, as part of a MSc thesis assignment
at the Eindhoven University of Technology (TU/e) - code and documentation may change.

## Overview

This repository contains all scripts and configuration files needed to run the experiments. To run 
them it is necessary to have installed the [Open AI Gym](https://gym.openai.com) and the 
[NasGym](https://github.com/gomerudo/nas-env). Also, the next repositories are required: [custom openai-baselines](https://github.com/gomerudo/openai-baselines/tree/experiments/baselines/meta_a2c), [nas-env](https://github.com/gomerudo/nas-env), and [custom meta-dataset](https://github.com/gomerudo/meta-dataset).

## Setup Instructions

We recommend to follow the next steps to avoid unnecessary changes in the scripts:

1. Clone all needed repositores into a directory `~/workspace/git_storage`.
2. Install mini-conda and setup an environment with name `nasdmrl` and Python 3.6.8
3. Run [scripts/setup/install_pypkgs.sh]()
4. Modify the paths in the nasenv.yml files under [configs/]()

## Run experiments

The experiments can be run with the [scripts/experiments/run_nas_dmrl.sh]() script.

## Authors

Student:
 - Jorge Gomez Robles (j.gomez.robles@student.tue.nl)

Supervisor:
 - Joaquin Vanschoren
