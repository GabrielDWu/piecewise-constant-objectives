import functools
import torch as th
from .algorithms import GMHP, exact_acc_rnn
from matplotlib import pyplot as plt
from tqdm import tqdm

full_names = {
    "acc": "Accuracy",
    "ce": "Cross Entropy",
    "ss": "Sigmoid Separation",
    "gmhp": "GMHP",
    "hook": "Hook Separation",
    "girard": "Girard Accuracy",
}


def lr_mult_schedule(step, *, n_steps, warmup_steps, decay_min_mult):
    warmup = min(step / warmup_steps, 1)
    decay = max(decay_min_mult ** (2 * step / n_steps), decay_min_mult)
    return warmup * decay

def train_model(model,
                objective="ce",
                num_steps=2000,
                track=["acc"],
                batch_size=2**20,
                C_gmhp=None,
                alpha_hook=0.1,
                show_plots=True,
                delta_ss=0.01,
                show_progress=True,
                
                lr_base=0.01,
                lr_warmup_ratio=.1,
                lr_decay_min_mult=0.01,
                ):
    """Train using a sharpened loss based on sigmoid of margin between correct and highest incorrect logit.
    
    Args:
        model: The RNN model to train
        objective: The method to train the model with. Can be "ce", "ss", "gmhp", "hook", or "girard"
        track: What to track during training. List, can include "acc" or any of the objectives
        C_gmhp: The pruning parameter for GMHP loss.
        alpha_hook: The thresholding parameter for hook loss.
        delta_ss: The scaling parameter for sigmoid separation loss.
        num_steps: Number of optimization steps
        batch_size: Batch size for training
        stop_early_patience: Number of steps to wait before stopping early if the loss is not improving. If None, will not stop early.
        show_progress: Whether to show a progress bar
        lr_base: Learning rate, scaled by sqrt(d)
        lr_warmup_ratio: Ratio of warmup steps to total steps
        lr_decay_min_mult: Minimum multiplier for learning rate decay
    """
    valid_objectives = ["ce", "ss", "gmhp", "hook"] + (["girard"] if model.n == 3 else [])
    valid_track = ["acc", "ce", "ss", "gmhp", "hook"] + (["girard"] if model.n == 3 else [])
    assert objective in valid_objectives, f"Invalid objective: {objective}"
    if objective not in track:
        track.append(objective)
    assert all(t in valid_track for t in track), f"Invalid track: {track}"

    lr = lr_base / model.d**0.5
    lr_lambda = functools.partial(
        lr_mult_schedule,
        n_steps=num_steps,
        warmup_steps=lr_warmup_ratio * num_steps,
        decay_min_mult=lr_decay_min_mult,
    )
    optimizer = th.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    losses = {t: [] for t in track}

    with th.enable_grad():
        for _ in tqdm(range(num_steps), disable=not show_progress):
            optimizer.zero_grad()

            if "gmhp" in track:
                gmhp = GMHP(model, C=C_gmhp)
                losses["gmhp"].append(gmhp.item())
                if "gmhp" == objective:
                    loss = -gmhp
            
            if "girard" in track:
                girard = exact_acc_rnn(model)
                losses["girard"].append(girard.item())
                if "girard" == objective:
                    loss = -girard

            x = th.randn(batch_size, model.n, device=model.device, dtype=model.dtype)
            y = x.argsort(dim=1)[:, -2]  # Second highest input is correct class
            logits = model(x)
            
            if "acc" in track:
                # For samples with all-zero logits, mark as incorrect
                pred = logits.argmax(dim=1)
                pred[(logits == 0).all(dim=1)] = -1

                accuracy = (pred == y).float().mean() + (pred == -1).float().mean() / model.n
                losses["acc"].append(accuracy.item())
            if "ce" in track:
                ce = th.nn.functional.cross_entropy(logits, y)
                losses["ce"].append(ce.item())
                if "ce" == objective:
                    loss = ce
            
            if "ss" in track or "hook" in track:
                correct_logits = logits[th.arange(batch_size), y]
                mask = th.ones_like(logits, dtype=th.bool)
                mask[th.arange(batch_size), y] = False
                max_incorrect_logits = logits[mask].reshape(batch_size, -1).max(dim=1).values
                margins = correct_logits - max_incorrect_logits
                if "ss" in track:
                    ss = th.sigmoid(-margins / delta_ss).mean()
                    losses["ss"].append(ss.mean().item())
                    if "ss" == objective:
                        loss = ss
                if "hook" in track:
                    hook = th.max(-margins + alpha_hook, th.zeros_like(margins)).mean()
                    losses["hook"].append(hook.item())
                    if "hook" == objective:
                        loss = hook
            
            loss.backward()
            
            # Check for NaN gradients before updating weights
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and th.isnan(param.grad).any():
                    has_nan_grad = True
                    print(f"NaN gradient detected in {name}")
                    
            if has_nan_grad:
                print("Model weights before NaN gradient step:")
                for name, param in model.named_parameters():
                    print(f"{name}: {param.data}")
                print("Returning early")
                return losses
                
            
            optimizer.step()
            lr_scheduler.step()
    
    # Create a single plot with all tracked metrics
    if show_plots:
        plot_losses(losses, objective, title=f"n={model.n}, d={model.d}, objective={full_names[objective]}")
    
    return losses

def plot_losses(losses, objective, title=""):
    # Count how many metrics we need to plot
    metrics_to_plot = list(losses.keys())
    n_metrics = len(metrics_to_plot)
    
    # Create a figure with vertical subplots - one for each method
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3*n_metrics), sharex=True)
    fig.suptitle(title, fontsize=16)
    
    # If there's only one metric, axes won't be an array
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric in its own subplot
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        label = full_names.get(metric, metric)
        
        if metric == objective:
            label += " (objective)"
            ax.plot(losses[metric], label=label, linewidth=2, color='red')
        else:
            ax.plot(losses[metric], label=label)
        
        ax.set_title(f"{label}")
        ax.grid(True, alpha=0.3)
    
    # Label the x-axis only on the bottom subplot
    axes[-1].set_xlabel('Gradient Steps')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for the suptitle
    plt.show()
