# DiffSRL: Learning Dynamic-Aware State representation for Control via Differentiable Simulation
## Pre-released code bug expected.

## Install
 - Run `cd ChamferDistancePytorch`
 - Install `python3 -m pip install -e .`
 - Run `cd ..`
 - Install `python3 -m pip install -e .`

## Enjoy the pretrained model
#### Model Free Reinforcement Learning on Chopsticks or Rope
- Run `python3 -m plb.algorithms.solve --algo td3 --env_name [Chopsticks-v1/Rope-v1] --exp_name enjoy --model_name rope/encoder --render`
#### Model Based Policy Optimization on Chopsticks or Rope
- Run `python3 -m plb.algorithms.solve --algo torch_nn --env_name [Chopsticks-v1/Rope-v1] --exp_name enjoy --model_name rope/encoder --render`

## Training new model
#### Collect data from new environment
- Run `python3 -m plb.algorithms.solve --algo collect --env_name [EnvName-version] --exp_name [new_environment]` Which will collect raw data and stored in `raw_data` folder.
- Run `python3 preprocess.py --dir raw_data/[new_environment]`to pre-process data and the preprocessed npz file will be stored in data with the name of `[new_environment]`.
#### Running State Representation learning using new dataset
- Run `python3 plb.algorithms.solve --env_name [EnvName-version] --exp_name [EnvName-version] --exp_name learn_latent  --lr 1e-5` The encoder weight will be saved in `pretrained_model`

## Experiment result
#### All experiment result are rendered from policy trained with MBPO
- Picking up a rope
![image](../images/chopsticks_srl.gif)

- Wrapping a rope around a cylinder
![image](../images/rope_srl.gif)

### TODO: Add more demos
