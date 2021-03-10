#!/usr/bin/env python
# Name   : test
# Author : Zhiao Huang
# Email  : z2huang@eng.ucsd.edu
# Date   : 2/13/2021
# Distributed under terms of the MIT license.

import os
import cv2
import numpy as np
import torch
import copy
import pickle
from typing import Optional, TypeVar, Type, Union

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
from plb.envs import make
env = make("Move-v1")
env.reset()

#%%
print(env.taichi_env.renderer.target_density.to_numpy().sum())
print(env.taichi_env.renderer.visualize_target)
plt.imshow(env.render(mode='rgb_array')[..., ::-1])
plt.show()

