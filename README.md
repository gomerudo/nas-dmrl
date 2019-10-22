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

[br_omni]: imgs/chained/average-best_reward-omniglot.png "Average best reward for omniglot"
[br_vgg]: imgs/chained/average-best_reward-vgg_flower.png "Average best reward for vgg_flower"
[br_dtd]: imgs/chained/average-best_reward-dtd.png "Average best reward for dtd"

[el_omni]: imgs/chained/average-ep_length-omniglot.png "Average episode length for omniglot"
[el_vgg]: imgs/chained/average-ep_length-vgg_flower.png "Average episode length for vgg_flower"
[el_dtd]: imgs/chained/average-ep_length-dtd.png "Average episode length for dtd"

[ar_omni]: imgs/chained/average-acc_reward-omniglot.png "Average accumulated reward for omniglot"
[ar_vgg]: imgs/chained/average-acc_reward-vgg_flower.png "Average accumulated reward for vgg_flower"
[ar_dtd]: imgs/chained/average-acc_reward-dtd.png "Average accumulated reward for dtd"


| Dataset    | Best reward  | Episode length | Acc. reward  |
|:----------:|:------------:|:--------------:|:------------:|
| `omniglot` | ![][br_omni] | ![][el_omni]   | ![][ar_omni] |
| `vgg_net`  | ![][br_vgg]  | ![][el_vgg]    | ![][ar_vgg]  |
| `dtd`      | ![][br_dtd]  | ![][el_dtd]    | ![][ar_dtd]  |

[en_train]: imgs/chained/entropy.png "Policy entropy during training"
[ad_train]: imgs/chained/actions-dist-training.png "Distribution of actions during training"

| Policy entropy  | Distribution of actions |
|:---------------:|:-----------------------:|
| ![][en_train]   | ![][ad_train]           |

### Experiment 2: evaluating the policy on previously unseen environments

#### a) Evaluatiing the policy 
[br_air]: imgs/chained/average-best_reward-aircraft.png "Average best reward for aircraft"
[br_cu]: imgs/chained/average-best_reward-cu_birds.png "Average best reward for cu_birds"

[el_air]: imgs/chained/average-ep_length-aircraft.png "Average episode length for aircraft"
[el_cu]: imgs/chained/average-ep_length-cu_birds.png "Average episode length for cu_birds"

[ar_air]: imgs/chained/average-acc_reward-aircraft.png "Average accumulated reward for aircraft"
[ar_cu]: imgs/chained/average-acc_reward-cu_birds.png "Average accumulated reward for cu_birds"

| Dataset    | Best reward  | Episode length | Acc. reward  |
|:----------:|:------------:|:--------------:|:------------:|
| `aircraft` | ![][br_air]  | ![][el_air]    | ![][ar_air]  |
| `cu_birds` | ![][br_cu]   | ![][el_cu]     | ![][ar_cu]   |

[ad_eval]: imgs/chained/actions-dist-evaluation.png "Distribution of actions during evaluation"

| Distribution of actions |
|:-----------------------:|
| ![][ad_eval]            |

#### b) Evaluating the designed networks

| Dataset    | Deep meta-RL (1st) | Deep meta-RL (2nd) | Shortened VGG19  |
|:----------:|:------------------:|:------------------:|:----------------:|
| `aircraft` | 49.18 ± 1.2        | **50.11 ± 1.02**   | 30.85 ± 10.82    |
| `cu_birds` | 23.97 ± 1.28       | **24.24 ± 0.90**   | 6.66 ± 1.98      |

### Experiment 3: training the meta-RL agent to design multi-branch structures

[br_omni-mb]: imgs/multibranch/average-best_reward-omniglot.png "Average best reward for omniglot (multi-branch)"
[el_omni-mb]: imgs/multibranch/average-ep_length-omniglot.png "Average episode length for omniglot (multi-branch)"
[ar_omni-mb]: imgs/multibranch/average-acc_reward-omniglot.png "Average accumulated reward for omniglot (multi-branch)"

| Dataset    | Best reward     | Episode length  | Acc. reward     |
|:----------:|:---------------:|:---------------:|:---------------:|
| `omniglot` | ![][br_omni-mb] | ![][el_omni-mb] | ![][ar_omni-mb] |

[ad_train-mb]: imgs/multibranch/actions-dist-mb.png "Distribution of actions during evaluation"

| Distribution of actions |
|:-----------------------:|
| ![][ad_train-mb]            |

## How to run

This repository (`nas-dmrl`) contains scripts to run the experiments. However, all the NAS and RL logic is exposed on independent repositories. We summarize all the requirements and main assumptions next.

### The directory structure

We assume the next directory structure:

```
${HOME}
└── workspace
    ├── git_storage
    │   ├── openai-baselines
    │   ├── meta-dataset
    │   └── nas-dmrl
    ├── logs
    ├── metadataset_storage
    │   └── records
    └── results
```

To download the openai-baselines

### Requirements

### Setup

We recommend to follow the next steps to avoid unnecessary changes in the scripts:

1. Create the workspace: `mkdir ${HOME}/workspace` and set the environment variable `WORKSPACE=${HOME}/workspace`
2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) into `${WORKSPACE}`, so that the miniconda path is `${WORKSPACE}/miniconda3`
3. Create the virtual environment `nasdmrl` with Python 3.6.8: `conda create -n nasdmrl python=3.6.8`
4. Run [scripts/setup/install_pypkgs.sh]() or install all packages listed there.
5. Install the [`nasgym`](https://github.com/gomerudo/nas-env)
6. Make sure that all meta-dataset files (TFRecords) are in `${WORKSPACE}/metadataset_storage/records`. If that is not the case, follow the instructions in [scripts/meta-dataset/README.md]() first.
7. Modify the paths in the all files under [configs/]() to match your `$HOME` directory and any other path you want to customize.

### Run experiments

#### Experiment 1

This first experiment requires to run an experiment per environment. Follow the next code snippets sequentially.

```
# Training on the omniglot environment
```

```
# Training on the vgg_flower environment
```

```
# Training on the dtd environment
```

The experiments can be run with the [scripts/experiments/run_nas_dmrl.sh]() script.

#### Experiment 2

#### Experiment 3
