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
rnn = RNN(hidden_size=3, seq_len=4, load_from_zoo=True)
losses = train_model(rnn, objective="gmhp", num_steps=250, lr_base=0.01, track=["acc"], C_gmhp=10000)
print(f"final gmhp loss: {losses['gmhp'][-1]:.4f}")
print(f"final acc: {losses['acc'][-1]:.4f}")
print(f"acc improvement: {losses['acc'][-1] - losses['acc'][0]:.4f}")
# %%
rnn = RNN(hidden_size=5, seq_len=8, load_from_zoo=True)
losses = train_model(rnn, objective="ss", num_steps=500, lr_base=0.01, track=["acc"], delta_ss=0.01, lr_decay_min_mult=0.1)
print(f"final ss loss: {losses['ss'][-1]:.4f}")
print(f"final acc: {losses['acc'][-1]:.4f}")
print(f"acc improvement: {losses['acc'][-1] - losses['acc'][0]:.4f}")
# %%
rnn = RNN(hidden_size=3, seq_len=4, load_from_zoo=True)
losses = train_model(rnn, objective="hook", num_steps=500, lr_base=0.01, track=["acc"], alpha_hook=1)
print(f"final hook loss: {losses['hook'][-1]:.4f}")
print(f"final acc: {losses['acc'][-1]:.4f}")
print(f"acc improvement: {losses['acc'][-1] - losses['acc'][0]:.4f}")
# %%
def run_model_grid(ns, ds, objectives={"gmhp": {"num_steps": 250, "C_gmhp": 10000}, "ss": {"num_steps": 500}, "hook": {"num_steps": 500, "alpha_hook": 1}}):
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
# %%
all_losses, final_accs = run_model_grid(ns, ds)
# %%
with open("piecewise-constant-objectives/data/all_losses_new.pkl", "rb") as f:
    all_losses = pickle.load(f)

final_accs = {}
for n in ns:
    final_accs[n] = {}
    for d in ds:
        final_accs[n][d] = {}
        for objective in all_losses[n][d]:
            final_accs[n][d][objective] = all_losses[n][d][objective]["acc"][-1]
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
            initial_acc = sampling_accuracy(rnn, n_test=2**24).item()
            
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
    y_limit = -0.07
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

# %%

def plot_objective_grid(all_losses, n, d):
    """
    Creates a 2x2 grid of subplots showing:
    - NW: Accuracy curves for all objectives (normalized by gradient steps)
    - NE: SS objective loss curve
    - SE: HH (Hook) objective loss curve
    - SW: GMHP objective loss curve
    
    Args:
        all_losses: Dictionary of losses indexed by n, d, and objective
        n: Sequence length
        d: Hidden size
    """
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle(f'n={n}, d={d}', fontsize=16)
    
    # Define colors for each objective (matching print_accuracy_table)
    colors = {
        'gmhp': '#1f77b4',  # blue
        'ss': '#ff7f0e',    # orange
        'hook': '#2ca02c'   # green
    }
    
    # Extract losses for the specified n and d
    losses = all_losses[n][d]
    
    # Get the number of steps for each objective
    gmhp_steps = len(losses['gmhp']['acc'])
    ss_steps = len(losses['ss']['acc'])
    hook_steps = len(losses['hook']['acc'])
    
    # NW subplot - Accuracy for all objectives (with normalized steps)
    ax_acc = axes[0, 0]
    
    # Create normalized x-axes for each accuracy curve
    gmhp_x = np.linspace(0, 1, gmhp_steps)
    ss_x = np.linspace(0, 1, ss_steps)
    hook_x = np.linspace(0, 1, hook_steps)
    
    # Plot accuracy curves
    ax_acc.plot(gmhp_x, losses['gmhp']['acc'], label='GMHP', color=colors['gmhp'], linewidth=2)
    ax_acc.plot(ss_x, losses['ss']['acc'], label='SS', color=colors['ss'], linewidth=2)
    ax_acc.plot(hook_x, losses['hook']['acc'], label='Hook', color=colors['hook'], linewidth=2)
    
    ax_acc.set_title('Accuracy')
    ax_acc.set_xlabel('Normalized Gradient Steps')
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()
    
    # NE subplot - SS objective
    ax_ss = axes[0, 1]
    if 'ss' in losses['ss']:  # The loss might be named 'ss'
        ax_ss.plot(losses['ss']['ss'], color=colors['ss'], linewidth=2)
    else:  # If there's no specific 'ss' loss, use the objective
        for key in losses['ss'].keys():
            if key != 'acc':  # Plot the non-accuracy loss
                ax_ss.plot(losses['ss'][key], color=colors['ss'], linewidth=2)
                break
    
    ax_ss.set_title('SS Objective')
    ax_ss.set_xlabel('Gradient Steps')
    ax_ss.grid(True, alpha=0.3)
    
    # SE subplot - Hook objective
    ax_hook = axes[1, 1]
    if 'hook' in losses['hook']:  # The loss might be named 'hook'
        ax_hook.plot(losses['hook']['hook'], color=colors['hook'], linewidth=2)
    else:  # If there's no specific 'hook' loss, use the objective
        for key in losses['hook'].keys():
            if key != 'acc':  # Plot the non-accuracy loss
                ax_hook.plot(losses['hook'][key], color=colors['hook'], linewidth=2)
                break
    
    ax_hook.set_title('Hook Objective')
    ax_hook.set_xlabel('Gradient Steps')
    ax_hook.grid(True, alpha=0.3)
    
    # SW subplot - GMHP objective
    ax_gmhp = axes[1, 0]
    if 'gmhp' in losses['gmhp']:  # The loss might be named 'gmhp'
        ax_gmhp.plot(losses['gmhp']['gmhp'], color=colors['gmhp'], linewidth=2)
    else:  # If there's no specific 'gmhp' loss, use the objective
        for key in losses['gmhp'].keys():
            if key != 'acc':  # Plot the non-accuracy loss
                ax_gmhp.plot(losses['gmhp'][key], color=colors['gmhp'], linewidth=2)
                break
    
    ax_gmhp.set_title('GMHP Objective')
    ax_gmhp.set_xlabel('Gradient Steps')
    ax_gmhp.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the suptitle
    plt.show()
# %%
for n in [4, 5, 6, 8]:
    for d in [3, 4, 5, 6]:
        if n == 4 and d == 3:
            continue
        plot_objective_grid(all_losses, n, d)

# %%