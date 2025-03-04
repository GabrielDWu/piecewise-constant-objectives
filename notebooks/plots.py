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
# %%

rnn = RNN(hidden_size=4, seq_len=5, load_from_zoo=True)
losses = train_model(rnn, objective="gmhp", num_steps=250, lr=0.0005, track=["acc"], C_gmhp=10000)
print(f"final gmhp loss: {losses['gmhp'][-1]:.4f}")
print(f"final acc: {losses['acc'][-1]:.4f}")
print(f"acc improvement: {losses['acc'][-1] - losses['acc'][0]:.4f}")
# %%
rnn = RNN(hidden_size=4, seq_len=4, load_from_zoo=True)
losses = train_model(rnn, objective="ss", num_steps=500, lr=0.005, track=["acc"], delta_ss=0.01)
print(f"final ss loss: {losses['ss'][-1]:.4f}")
print(f"final acc: {losses['acc'][-1]:.4f}")
print(f"acc improvement: {losses['acc'][-1] - losses['acc'][0]:.4f}")
# %%
rnn = RNN(hidden_size=4, seq_len=5, load_from_zoo=True)
losses = train_model(rnn, objective="hook", num_steps=500, lr=0.005, track=["acc"], alpha_hook=1)
print(f"final hook loss: {losses['hook'][-1]:.4f}")
print(f"final acc: {losses['acc'][-1]:.4f}")
print(f"acc improvement: {losses['acc'][-1] - losses['acc'][0]:.4f}")
# %%
def run_model_grid(ns, ds, objectives={"gmhp": {"num_steps": 250, "C_gmhp": 10000, "lr": 0.0005}, "ss": {"num_steps": 500, "lr": 0.005}, "hook": {"num_steps": 500, "alpha_hook": 1, "lr": 0.005}}):
    """
    Generates a dictionary all_losses[n][d][objective], and final_accs[n][d][objective]
    """
    all_losses = {n: {d: {} for d in ds} for n in ns}
    final_accs = {n: {d: {} for d in ds} for n in ns}

    import itertools
    from tqdm import tqdm
    
    for n, d, (objective, params) in tqdm(list(itertools.product(ns, ds, objectives.items())), desc="run_model_grid"):
        rnn = RNN(hidden_size=d, seq_len=n, load_from_zoo=True)
        losses = train_model(rnn, objective=objective, track=["acc"], **params, show_progress=False)
        all_losses[n][d][objective] = losses
        final_accs[n][d][objective] = losses["acc"][-1]
        print(f"n={n}, d={d}, objective={objective}, final acc={final_accs[n][d][objective]}; Δ(acc)={losses['acc'][-1] - losses['acc'][0]}")

    return all_losses, final_accs

# %%
ns = [4, 5, 6, 8]
ds = [3, 4, 5, 6]
all_losses, final_accs = run_model_grid(ns, ds)
# %%
import pickle
# save all_losses to a pickle file
with open("piecewise-constant-objectives/notebooks/all_losses.pkl", "wb") as f:
    pickle.dump(all_losses, f)
# %%

def print_accuracy_table(ns, ds, final_accs):
    """
    Prints a table with columns of n, d, initial accuracy, and accuracy differential from each method.
    Also generates a bar plot comparing the accuracy differentials across methods.
    
    Args:
        ns: List of sequence lengths
        ds: List of hidden sizes
        final_accs: Dictionary of final accuracies indexed by n, d, and objective
    """
    print(f"{'n':^5}|{'d':^5}|{'Initial Acc':^12}|{'GMHP Diff':^12}|{'SS Diff':^12}|{'Hook Diff':^12}")
    print("-" * 65)
    
    from piecewise_constant_objectives.algorithms import sampling_accuracy
    
    # Store the data for plotting
    all_nd_labels = []
    gmhp_diffs = []
    ss_diffs = []
    hook_diffs = []
    
    for n in ns:
        for d in ds:
            # Create a new RNN instance for initial accuracy calculation
            rnn = RNN(hidden_size=d, seq_len=n, load_from_zoo=True)
            
            # Calculate initial accuracy using sampling_accuracy
            initial_acc = sampling_accuracy(rnn, n_test=2**24)
            
            # Calculate accuracy differentials for each method
            gmhp_diff = final_accs[n][d]["gmhp"] - initial_acc
            ss_diff = final_accs[n][d]["ss"] - initial_acc
            hook_diff = final_accs[n][d]["hook"] - initial_acc
            
            # Store the data for plotting
            all_nd_labels.append(f"({n},{d})")
            gmhp_diffs.append(gmhp_diff)
            ss_diffs.append(ss_diff)
            hook_diffs.append(hook_diff)
            
            # Print the row
            print(f"{n:^5}|{d:^5}|{initial_acc:^12.4f}|{gmhp_diff:^12.4f}|{ss_diff:^12.4f}|{hook_diff:^12.4f}")
    
    # Create a bar plot for the accuracy differentials
    plt.figure(figsize=(15, 4))
    
    # Set up the x-axis positions for the grouped bars
    x = np.arange(len(all_nd_labels))
    width = 0.25  # Width of the bars
    
    # Create the bars for each method
    plt.bar(x - width, gmhp_diffs, width, label='GMHP', color='#1f77b4')
    plt.bar(x, ss_diffs, width, label='SS', color='#ff7f0e')
    plt.bar(x + width, hook_diffs, width, label='Hook', color='#2ca02c')
    
    # Set the y-axis limit
    y_limit = -0.04
    plt.ylim(bottom=y_limit)
    
    # Add value labels for hook_diff bars that are cut off by the y-axis limit
    for i, value in enumerate(hook_diffs):
        if value < y_limit:
            plt.text(x[i] + width, y_limit + 0.025, f"{value:.2g}".replace("-","—"), ha='center', va='top', fontsize=8, rotation=90)
    
    # Add value labels for ss_diff bars that are cut off by the y-axis limit
    for i, value in enumerate(ss_diffs):
        if value < y_limit:
            plt.text(x[i], y_limit + 0.025, f"{value:.2g}".replace("-","—"), ha='center', va='top', fontsize=8, rotation=90)

    # Add labels, title, and legend
    plt.xlabel('RNN size (n, d)')
    plt.ylabel('Accuracy Improvement')
    plt.xticks(x, all_nd_labels)
    plt.legend()
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Call the function with our ns, ds, and final_accs
print_accuracy_table(ns, ds, final_accs)

# %%
# Train on perfect accuracy
rnn = RNN(hidden_size=4, seq_len=3, load_from_zoo=True)
# losses = train_model(rnn, objective="girard", num_steps=250, lr=0.0005, track=["ce", "acc"])
# print(f"final girard acc: {losses['girard'][-1]:.4f}")
# print(f"final sampling acc: {losses['acc'][-1]:.4f}")
# print(f"girard acc improvement: {losses['girard'][-1] - losses['girard'][0]:.4f}")
# %%
exact_acc_rnn(rnn)
# %%
sampling_accuracy(rnn, n_test=2**24)