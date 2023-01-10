# Dual adaptive training of photonic neural networks
This repository is for dual adaptive training (DAT) of inference-based photonic neural networks (MPNNs) proposed in the following paper:

Ziyang Zheng*, Zhengyang Duan*, Hang Chen, Rui Yang, Sheng Gao, Haiou Zhang, Hongkai Xiong, Xing Lin, "Dual adaptive training of photonic neural networks." arXiv preprint arXiv:2212.06141 (2022). (*These Authors contributed equally to this work. DOI: https://doi.org/10.48550/arXiv.2212.06141)

The codes for the construction of MPNNs and the complex-valued SEPNs are based on the Neurophox repository (https://github.com/solgaardlab/neurophox) and (https://github.com/wavefrontshaping/complexPyTorch), respectively.

<!-- vim-markdown-toc GFM -->

* [Introduction](#introduction)
* [Dependencies](#dependencies)
* [Running codes](#running-codes)
* [Generating figures in the paper](#generating-figures-in-the-paper)

<!-- vim-markdown-toc -->

## Introduction
Photonic neural network (PNN) is a remarkable analog artificial intelligence (AI) accelerator that computes with photons instead of electrons to feature low latency, high energy efficiency, and high parallelism. However, the existing training approaches cannot address the extensive accumulation of systematic errors in large-scale PNNs, resulting in a significant decrease in model performance in physical systems. Here, we propose dual adaptive training (DAT) that allows the PNN model to adapt to substantial systematic errors and preserves its performance during the deployment. By introducing the systematic error prediction networks with task-similarity joint optimization, DAT achieves the high similarity mapping between the PNN numerical models and physical systems and high-accurate gradient calculations during the dual backpropagation training. We validated the effectiveness of DAT by using diffractive PNNs and interference-based PNNs on image classification tasks. DAT successfully trained large-scale PNNs under major systematic errors and preserved the model classification accuracies comparable to error-free systems. The results further demonstrated its superior performance over the state-of-the-art in situ training approaches. DAT provides critical support for constructing large-scale PNNs to achieve advanced architectures and can be generalized to other types of AI systems with analog computing errors.

## Dependencies
Some important dependencies are specified in `requirements.txt` as follows:
```text
numpy>=1.16
scipy
tensorflow>=2.2.0
torch>=1.10
```

## Running codes

To train (or test) a 3-layer MPNN with 64-dimensional input for MNIST classification in an error-free system, use the following command:
```
python main.py -g 0 -wave_dims 64 -train (-test) -id 0 \
    -iterN 3 -method ideal -epoch 50 -bs 500 -dataN MNIST 
```

Explanation for the options (all options are parsed in `config.py`):
* `-g/--gpu`: the id of GPU used. GPU 0 will be used by default.
* `-wave_dims/--waveguide_dims`: the dimension of input optical field.
* `-train (-test)` option indicates training or testing mode. 
* `-id/--exp_id`: experiment id, used to differentiate experiments with the same setting.
* `-iterN/--iter_name`: the number of photonic meshes for constructing MPNN.
* `--method/--train_method`: the method to train the MPNN, including:
  * `ideal` standing for in silico training in an error-free system;
  * `pat` for physics-aware training;
  * `dat` for dual adaptive training.
* `-epoch`: the epoches in training stage.
* `-bs/--train_batch_size`: the batch size in training stage.
* `-dataN/--data_name`: names of dataset, including MNIST and FMNIST.


To train (or test) the MPNN with PAT under systematic errors, use the following command:
```
python main.py -g 0 -wave_dims 64 -train (-test) -id 0 \
    -iterN 3 -method pat -epoch 50 -bs 500 -dataN MNIST \
    -bsE 0.04 -phaseE 0.04 -Etype bothE 
```

Explanation for the new options:
* `-bsE/--bs_error`: std of beamspliter errors.
* `-phaseE/--phase_error`: std of phase shifter errors.
* `-Etype/--error_type`: choices: ['noE', 'phaseE', 'bsE', 'bothE']: 
  * `noE` standing for the error-free system;
  * `phaseE` for the individual phase shifter errors;
  * `bsE` for individual beamsplitter errors;
  * `bothE` for joint errors.

To train (or test) the MPNN with DAT under systematic errors, use the following command:
```
python main.py -g 0 -wave_dims 64 -train (-test) -id 0 \
    -iterN 3 -method dat -epoch 50 -bs 500 -dataN MNIST \
    -phaseE 0.04 -Etype phaseE -cn UNet -ks 3 -FM_num 4 8 16 
    (-misu) (-miss) (-p)
```
Explanation for the new options:
* `-cn/--cn_model`: the choice of SEPN. Default is UNet,
* `-ks/--kernel_size`: kernel size of convolution layers of SEPNs.
* `-FM_num`: number of feature maps, refer to F_1, F_2 and F_3 in the paper.
* `-misu/--meas_IS_unitary (-miss/--meas_IS_separable)`: whether implementing DAT in unitary or separable mode. `-misu` for unitary mode and `-miss` for separable mode.
* `-p/--plot`: the options for saving phase, confusion matrix and energe distribution during testing phase.

##  Generating figures in the paper

All the models for generating Fig.4(b), 4(c), 4(d), 4(e), 5(c) are saved in `./logs`.

To generate the result for Direct Deployment in Fig.4(b) when std of phase shifter error is 0.04, use the following command:

```
python main.py -g 0 -wave_dims 64 -test -id 0 -iterN 3 -method ideal -bs 500 -dataN MNIST -phaseE 0.04 -Etype phaseE 
```

To generate the result for PAT in Fig.4(c) when std of beamsplitter error is 0.06, use the following command:

```
python main.py -g 0 -wave_dims 64 -test -id 0 -iterN 3 -method pat -bs 500 -dataN FMNIST -bsE 0.06 -Etype bsE 
```

To generate the result for DAT w/o IS (~4k Params) in Fig.4(d) when std of beamsplitter error is 0.02, use the following command:

```
python main.py -g 0 -wave_dims 64 -test -id 0 -iterN 3 -method dat -bs 500 -dataN MNIST -bsE 0.02 -Etype bsE -cn UNet -ks 3 -FM_num 4 6 8
```

To generate the result for DAT w/ IS (~10k Params) in Fig.4(d) when std of phase shifter error is 0.08, use the following command:

```
python main.py -g 0 -wave_dims 64 -test -id 0 -iterN 3 -method dat -bs 500 -dataN MNIST -phaseE 0.08 -Etype phaseE -cn UNet -ks 3 -FM_num 4 8 16 -misu
```

To generate the result for DAT w/ IS Sp (~10k Params) in Fig.4(e) when std of phase shifter error is 0.04, use the following command:

```
python main.py -g 0 -wave_dims 64 -test -id 0 -iterN 3 -method dat -bs 500 -dataN MNIST -phaseE 0.04 -Etype phaseE -cn UNet -ks 3 -FM_num 4 8 16 -miss
```

To generate the result for DAT w/ IS (~10k Params) in Fig.5(c) when stds of beamsplitter and phase shifter errors are 0.06 and 0.06 in the task of MNIST classification, use the following command:

```
python main.py -g 0 -wave_dims 64 -test -id 0 -iterN 3 -method dat -bs 500 -dataN MNIST -bsE 0.06 -phaseE 0.06 -Etype bothE -cn UNet -ks 3 -FM_num 4 8 16 -misu -p
```
