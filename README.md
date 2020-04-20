# PyTorch AWR
This repository contains a [PyTorch](https://pytorch.org/) implementation of the reinforcement learning algorithm [Advantage Weighted Regression (AWR)](https://arxiv.org/abs/1910.00177).
The objective of this implementation is to make it possible for PyTorch users to use AWR for their RL projects, as the original implementation is for TensorFlow (see references).

## Setup
- make sure you are running Python 3.6.9 or above
- run `pip3 install -r requirements.txt --no-cache-dir` (the `no-cache-dir`-option is sometimes required to finish the download of `torch`)
- you can remove `mujoco-py` from the requirements if you do not have a license
- edit `main.py` to configure your environment and hyper-parameters (cli options are planned)
- run `pyton3 main.py`

## Features
- full implementation of AWR according to the paper
- hyper-parameters pre-filled with appropriate values
- training and testing framework: given the NN models, the environment and the hyper-parameters, the framework trains the models and conducts a series of tests on them after completion 

## References
The authors' code (written for [TensorFlow](https://www.tensorflow.org/)) for the paper can be found [here](https://github.com/xbpeng/awr).