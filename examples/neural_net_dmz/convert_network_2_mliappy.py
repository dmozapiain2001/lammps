import sys
print(sys.path)

import glob
import os

import numpy as np






import sys
import timeit

import shutil
from numpy import *


import re

# import matplotlib.pyplot as plt
# %matplotlib inline



import os
import pandas as pd
# import matplotlib.pyplot as plt

import csv



# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split


# from sklearn.metrics import mean_squared_error, mean_absolute_error





import glob

# import xmltodict

import glob
import torch
import time
import numpy as np
import torch.nn as nn
import scipy.io as sio
# import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.ndimage as ndimage
from torchvision import transforms, utils
from toolz.curried import pipe, curry, compose
from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split


# # Neural Network
#
# Real simple neural network with 1 hidden layer

# In[22]:

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path_f=os.getcwd()

folder=path_f+'/neural_net_weights_per_atom_energy_1_hidden_layer_optimization_wrapper_batch_opt'

model_state = torch.load(folder+"/learnt_weights_C0_trial_1_1000", map_location=torch.device("cpu"))
# atom_network.load_state_dict(model_state)

atom_network2= torch.nn.Sequential(
    torch.nn.Linear(30,6),
    torch.nn.ReLU(),
    torch.nn.Linear(6,1))


with torch.no_grad():
    atom_network2[0].weight=nn.Parameter(model_state['atom_network.0.weight'])
    atom_network2[0].bias=nn.Parameter(model_state['atom_network.0.bias'])

    atom_network2[2].weight=nn.Parameter(model_state['atom_network.2.weight'])
    atom_network2[2].bias=nn.Parameter(model_state['atom_network.2.bias'])




model_lin = atom_network2
pytorch_total_params = np.sum(p.numel() for p in model_lin.parameters() if p.requires_grad)





import sys
import numpy as np
import torch
import pickle
import os
import shutil



shutil.copyfile('../../src/MLIAPPY/mliappy_pytorch.py','./mliappy_pytorch.py')

import mliappy_pytorch

model = mliappy_pytorch.IgnoreElems(model_lin)
n_descriptors = 30
n_params = mliappy_pytorch.calc_n_params(model)
n_types = 1
linked_model = mliappy_pytorch.TorchWrapper64(model,n_descriptors=n_descriptors,n_elements=n_types)

# Save the result.
with open("Ta06A.mliap.pytorch.model.pkl",'wb') as pfile:
    pickle.dump(linked_model,pfile)

# torch.save(linked_model,"Ta06A.mliap.pytorch.model.pkl")

