# Learning to reinforcement learn for Neural Architecture Search

MSc thesis project developed by Jorge Gomez Robles
at the Eindhoven University of Technology (TU/e), under the supervision of dr. ir. Joaquin Vanschoren.

You can check the paper [here](files/L2RL-NAS.pdf).


## Points of contact

 - Jorge Gomez Robles (j.gomezrb.dev@gmail.com)
 - Joaquin Vanschoren (j.vanschoren@gmail.com)

## Overview of the research project

The ultimate goal of Neural Architecture Search (NAS) is to come up with an algorithm that can design well-performing architectures for any dataset of interest. A promising approach to NAS is reinforcement learning (RL). 

One of the main limitations of RL on the NAS problem is the need to run the procedure from scratch for every dataset of interest. So far, most of the relevant results show how to apply standard RL algorithms on NAS for CIFAR, but little attention is paid to other datasets. Moreover, RL tends to be an expensive procedure for NAS, making it unfeasible to replay it on new datasets.

An alternative is to explore meta-RL, which can learn a policy that can be transferred to previously unseen environments (i.e., datasets). In this work, we explore, for the first time, meta-RL for NAS. We study whether or not the transfer provides an advantage during training and evaluation (i.e., when the policy is fixed).

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

### Setup

Follow the next steps to avoid unnecessary changes in the scripts:

1. Create the workspace: `mkdir ${HOME}/workspace` and set the environment variable `WORKSPACE=${HOME}/workspace`
2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) into `${WORKSPACE}`, so that the miniconda path is `${WORKSPACE}/miniconda3`
3. Create the virtual environment `nasdmrl` with Python 3.6.8: `conda create -n nasdmrl python=3.6.8`
4. Run [scripts/setup/install_pypkgs.sh]() or install all packages listed there.
5. Install the [`nasgym`](https://github.com/gomerudo/nas-env)
6. Make sure that all meta-dataset files (TFRecords) are in `${WORKSPACE}/metadataset_storage/records`. If that is not the case, follow the instructions in [scripts/meta-dataset/README.md]() first.
7. Modify the paths in the all files under [configs/](configs/) to match your `$HOME` directory and any other path you want to customize.

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

### Run experiments

#### Experiment 1

This first experiment requires to run an experiment per environment. Follow the next code snippets sequentially.

##### Deep meta-RL

```sh
# Training on the omniglot environment

${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-training/run_nas_dmrl.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/meta-rl/config-omniglot.ini \
  -r > omniglot.log

# The log contains the directory where all the OpenAI logs are stored, there the policy is stored. 
# You can find it with the next command:

cat omniglot.log | grep "Saving trained model"
# Expected path is something like:
# /home/jgomes/workspace/results/experiment-20190816164503/openai-20190816164503/models/meta_a2c_final.model
```

```sh
# Training on the vgg_flower environment

# Assuming that we rename the policy from omniglot as 
# ${HOME}/workspace/results/policy-omniglot.model
${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-training/run_nas_dmrl.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/meta-rl/config-vgg_flower.ini \
  -m ${HOME}/workspace/results/policy-omniglot.model \
  -r > vgg_flower.log
```

```sh
# Training on the dtd environment

# Assuming that we rename the policy from vgg_flower as 
# ${HOME}/workspace/results/policy-vgg_flower.model
${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-training/run_nas_dmrl.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/meta-rl/config-dtd.ini \
  -m ${HOME}/workspace/results/policy-vgg_flower.model \
  -r > dtd.log
```

##### DeepQN

```sh
# Training on the omniglot environment

${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-benchmarks/run_nas_dqn.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/dqn/config-omniglot.ini \
  -r > omniglot-dqn.log
```

```sh
# Training on the vgg_flower environment

${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-benchmarks/run_nas_dqn.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/dqn/config-vgg_flower.ini \
  -r > vgg_flower-dqn.log
```

```sh
# Training on the dtd environment

${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-benchmarks/run_nas_dqn.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/dqn/config-dtd.ini \
  -r > dtd-dqn.log
```

##### Random search

```sh
# Running on the omniglot environment

${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-benchmarks/run_nas_random.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/random-search/config-omniglot.ini \
  -r > omniglot-rs.log
```

```sh
# Running on the vgg_flower environment

${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-benchmarks/run_nas_random.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/random-search/config-vgg_flower.ini \
  -r > vgg_flower-rs.log
```

```sh
# Running on the dtd environment

${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-benchmarks/run_nas_random.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/random-search/config-dtd.ini \
  -r > dtd-rs.log
```

#### Experiment 2

##### Evaluation of the policy

```sh
# Evaluating on aircraft

# Assuming the final policy from the Experiment 1 is renamed as 
# ${HOME}/workspace/results/policy-dtd.model
${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-evaluation/evaluate_nas_dmrl.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/meta-rl/config-aircraft.ini \
  -m ${HOME}/workspace/results/policy-dtd.model \
  -r > aircraft.log
```

```sh
# Evaluating on cu_birds

# Assuming the final policy from the Experiment 1 is renamed as 
# ${HOME}/workspace/results/policy-dtd.model
${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-evaluation/evaluate_nas_dmrl.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/meta-rl/config-cu_birds.ini \
  -m ${HOME}/workspace/results/policy-dtd.model \
  -r > cu_birds.log
```

##### Random search on evaluation datasets
```sh
# Running on the aircraft environment

${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-benchmarks/run_nas_random.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/random-search/config-aircraft.ini \
  -r > aircraft-rs.log
```

```sh
# Running on the cu_birds environment

${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-benchmarks/run_nas_random.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/random-search/config-cu_birds.ini \
  -r > cu_birds-rs.log
```

##### Evaluating the best 2 architectures per dataset

To obtain the best two architectures per dataset, you can query the `/home/jgomes/workspace/logs/dmrl/db_experiments.csv` file, which is set in `${WORKSPACE}/git_storage/nas-dmrl/configs/meta-rl/config-cu_birds.ini` and `${WORKSPACE}/git_storage/nas-dmrl/configs/meta-rl/config-aircraft.ini`. For more information, check the `nasgym` documentation.

Once that the best architectures have been identified, go to the `${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-evaluation/network_evaluation.py` and manually enter the architecture as a list of NSCs. Now, you can run the next command per network:

```
# Change the dataset's config file accordingly.
${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-evaluation/run_network_evaluation.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/meta-rl/config-aircraft.ini \
  -r > aircraft-network.log
```

##### Train the shortened version of the VGG19

```
# Change the dataset's config file accordingly.
${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-benchmarks/run_network_benchmarking.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/meta-rl/config-aircraft.ini \
  -r > aircraft-network.log
```


#### Experiment 3

The third experiment is similar to experiment 1, but with only one environment and in a multi-branch setting. Run the next commands to obtain the results:

```sh
# Training on the omniglot environment with sigma=0.0

${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-training/run_nas_dmrl.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/meta-rl/config-omniglot-mb-00.ini \
  -r > omniglot-mb-00.log
```

```sh
# Training on the omniglot environment with sigma=0.1

${WORKSPACE}/git_storage/nas-dmrl/scripts/experiments-training/run_nas_dmrl.sh \
  -c ${WORKSPACE}/git_storage/nas-dmrl/configs/meta-rl/config-omniglot-mb-10.ini \
  -r > omniglot-mb-10.log
```

## How to visualize the results

To visualize the plots summarizing the results, use the notbookes in [notebooks/](notebooks/).