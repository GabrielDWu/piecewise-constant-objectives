# %%
# type: ignore[syntax]
%load_ext autoreload
# type: ignore[syntax]
%autoreload 2

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch as th
from matplotlib import pyplot as plt
import numpy as np
from piecewise_constant_objectives import *
import pickle

# %%
rnn3 = RNN(hidden_size=3, seq_len=3, load_from_zoo=True)
losses3 = train_model(rnn3, objective="girard", num_steps=500, lr_base=0.005, lr_decay_min_mult=1, track=["ce", "acc"])
losses3short = {}
for key in losses3.keys():
    if key != "acc":
        losses3short[key] = losses3[key][:301]

# This is a Figure.
plot_losses(losses3short, "girard", title="n=3, d=3")

# %%
## Switch accuracy to Acc'. Then, this is a Figure.
rnn = RNN(hidden_size=2, seq_len=3, load_from_zoo=True)
losses = train_model(rnn, objective="girard", num_steps=2000, lr_base=0.005 * 1.4, lr_decay_min_mult=1, track=["ce", "acc"])
