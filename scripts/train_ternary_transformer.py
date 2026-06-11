import torch
import torch.nn.functional as F
import os
import time
import argparse
from tqdm import tqdm
import numpy as np
from config.config import default_config as config
from src.models.transformer import TernaryTransformer
from data_loader.data_loader import get_batch_iterator
from typing import Dict


# --- Runtime Diagnostics Helpers ---

def bytes_to_gib(num_bytes: int) -> float:
    """Convert a byte count to gibibytes for human-readable memory reports."""
    return num_bytes / (1024 ** 3)


def get_device_report(device: str) -> str:
    """
    Build a short report describing the runtime environment: PyTorch/CUDA
    versions and, when running on a GPU, its name, capability, and total VRAM.
    This makes it easy to collect comparable training reports across machines.
    """
    lines = [
        f"PyTorch version: {torch.__version__}",
        f"Configured device: {device}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"CUDA version: {torch.version.cuda}",
    ]

    if device.startswith('cuda') and torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        total_vram_gib = bytes_to_gib(props.total_memory)
        lines.extend([
            f"GPU name: {torch.cuda.get_device_name(device_index)}",
            f"GPU capability: {props.major}.{props.minor}",
            f"Total VRAM: {total_vram_gib:.2f} GiB",
        ])
    else:
        lines.append("GPU name: N/A (running without CUDA)")

    return "\n".join(lines)


def get_peak_memory_report(device: str) -> str:
    """Report peak GPU memory (allocated/reserved) since the last reset, or N/A on CPU."""
    if device.startswith('cuda') and torch.cuda.is_available():
        peak_allocated = bytes_to_gib(torch.cuda.max_memory_allocated())
        peak_reserved = bytes_to_gib(torch.cuda.max_memory_reserved())
        return (
            f"Peak VRAM allocated: {peak_allocated:.2f} GiB | "
            f"Peak VRAM reserved: {peak_reserved:.2f} GiB"
        )
    return "Peak VRAM allocated: N/A | Peak VRAM reserved: N/A"


def format_zone_distribution(zones: Dict[str, int]) -> str:
    """Format a zone distribution dict into a human-readable string."""
    return f"Green={zones.get('GREEN', 0)}  Yellow={zones.get('YELLOW', 0)}  Red={zones.get('RED', 0)}"


def format_trace(step: int, trace: list) -> str:
    """Format the full decision trace for a given step into a compact log string."""
    lines = [f"--- Trace at step {step} ---"]
    for block_entry in trace:
        lines.append(f"  {block_entry}")
    return "\n".join(lines)


# --- Argument Parsing ---

parser = argparse.ArgumentParser(description="Train a TernaryTransformer with conservation loss.")
parser.add_argument(
    "--ternary", action="store_true", default=False,
    help="Enable ternary weights (TernaryTransformer with conservation loss, confidence zones, and decision traces). "
         "Defaults to False (standard Transformer). This script always uses TernaryTransformer; "
         "this flag is kept for interface consistency."
)
parser.add_argument(
    "--trace-freq", type=int, default=5000,
    help="Log decision trace and head-zone distribution every N steps (default: 5000)."
)
parser.add_argument(
    "--conservation-coeff", type=float, default=0.01,
    help="Coefficient for conservation loss (default: 0.01)."
)
args = parser.parse_args()

# Override config with CLI flags
use_ternary = args.ternary or config.get('ternary_weights', False)
trace_freq = args.trace_freq
conservation_coeff = args.conservation_coeff

# --- Initialize the Model and Print Parameters ---

# Print runtime/device diagnostics and reset GPU peak-memory stats before training.
print(get_device_report(config['device']))
if config['device'].startswith('cuda') and torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

model = TernaryTransformer(
    n_head=config['n_head'],
    n_embed=config['n_embed'],
    context_length=config['context_length'],
    vocab_size=config['vocab_size'],
    N_BLOCKS=config['n_blocks']
).to(config['device'])

# Print the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params:,}")
print(f"Ternary weights: {use_ternary} | Conservation coefficient: {conservation_coeff} | Trace frequency: {trace_freq}")

# Print initial ternary diagnostics
initial_confidence = model.confidence
initial_zones = model.confidence_distribution()
initial_head_zones = model.head_zone_distribution()
print(f"Initial model confidence: {initial_confidence:.4f}")
print(f"Initial block zone distribution: {format_zone_distribution(initial_zones)}")
print(f"Initial head zone distribution:  {format_zone_distribution(initial_head_zones)}")

# --- Optimizer Setup and Loss Tracking ---

# Set up the AdamW optimizer with the specified learning rate.
optimizer = torch.optim.AdamW(model.parameters(), lr=config['t_lr'])

# List to track loss values during training.
losses = []

# Define a window size for averaging recent losses in the training loop.
AVG_WINDOW = 64

# Helper function to estimate the average loss for training and development data.
@torch.no_grad()
def estimate_loss(steps: int) -> Dict[str, float]:
    """
    Evaluate the model on training and development datasets and calculate average loss.

    Args:
        steps (int): Number of steps to evaluate.

    Returns:
        dict: Dictionary containing average losses for 'train' and 'dev' splits.
    """
    out = {}
    model.eval()  # Set the model to evaluation mode.

    for split in ['train', 'dev']:
        # Select the appropriate data path for the current split.
        data_path = config['train_path'] if split == 'train' else config['dev_path']

        # Create a batch iterator for evaluation.
        batch_iterator_eval = get_batch_iterator(
            data_path, config['t_batch_size'], config['t_context_length'], device=config['device']
        )

        # Initialize a tensor to track loss values for each evaluation step.
        losses_eval = torch.zeros(steps)
        for k in range(steps):
            try:
                # Fetch a batch and calculate the loss.
                xb, yb = next(batch_iterator_eval)
                _, loss = model(xb, yb)
                losses_eval[k] = loss.item()
            except StopIteration:
                # Handle the case where the data iterator ends early.
                print(f"Warning: Iterator for {split} ended early.")
                break

        # Compute the mean loss for the current split.
        out[split] = losses_eval[:k + 1].mean()

    model.train()  # Restore the model to training mode.
    return out

# --- Training Loop ---

# Create a batch iterator for the training data.
batch_iterator = get_batch_iterator(
    config['train_path'],
    config['t_batch_size'],
    config['t_context_length'],
    device=config['device']
)

# Number of tokens processed per step (batch_size * context_length), used for throughput.
tokens_per_step = config['t_batch_size'] * config['t_context_length']
last_eval_time = time.perf_counter()

# Create a progress bar to monitor training progress.
pbar = tqdm(range(config['t_train_steps']))
for step in pbar:
    try:
        # Fetch a batch of input and target data (and start the step timer).
        step_start_time = time.perf_counter()
        xb, yb = next(batch_iterator)

        # Perform a forward pass and compute the loss.
        _, loss = model(xb, yb)

        # Add conservation loss for ternary model
        if use_ternary:
            conservation = model.get_conservation_loss()
            total_loss = loss + conservation_coeff * conservation
        else:
            conservation = torch.tensor(0.0, device=config['device'])
            total_loss = loss

        # Record the loss for tracking.
        losses.append(total_loss.item())
        pbar.set_description(f"Train loss: {np.mean(losses[-AVG_WINDOW:]):.4f}")

        # Backpropagate the loss and update the model parameters.
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()

        # Clip gradients to prevent exploding gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Measure step time and instantaneous throughput for diagnostics.
        step_time = time.perf_counter() - step_start_time
        tokens_per_second = tokens_per_step / step_time if step_time > 0 else float('inf')

        # Periodically evaluate the model on training and development data.
        if step % config['t_eval_steps'] == 0:
            evaluation_losses = estimate_loss(config['t_eval_iters'])
            train_loss = evaluation_losses['train']
            dev_loss = evaluation_losses['dev']
            # Report timing/throughput for the most recent step and wall-time since last eval.
            now = time.perf_counter()
            elapsed_since_eval = now - last_eval_time
            last_eval_time = now

            # Build ternary-specific diagnostics line
            ternary_diag = ""
            if use_ternary:
                conf = model.confidence
                b_zones = model.confidence_distribution()
                h_zones = model.head_zone_distribution()
                ternary_diag = (
                    f", Confidence: {conf:.4f}, "
                    f"Blocks [{format_zone_distribution(b_zones)}], "
                    f"Heads [{format_zone_distribution(h_zones)}], "
                    f"Conservation: {conservation.item():.6f}"
                )

            print(
                f"Step: {step}, Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}, "
                f"Step time: {step_time:.3f}s, Throughput: {tokens_per_second:.2f} tokens/s, "
                f"Elapsed since last eval: {elapsed_since_eval:.2f}s{ternary_diag}"
            )
            print(get_peak_memory_report(config['device']))

        # Log ternary decision trace every trace_freq steps
        if use_ternary and step > 0 and step % trace_freq == 0:
            # Trace info: confidence, zone distribution, and full decision trace
            current_confidence = model.confidence
            current_zones = model.confidence_distribution()
            current_head_zones = model.head_zone_distribution()
            current_conservation = conservation.item()

            print(f"\n=== Ternary Trace | Step {step} ===")
            print(f"  Model confidence: {current_confidence:.4f}")
            print(f"  Block zones:      {format_zone_distribution(current_zones)}")
            print(f"  Head zones:       {format_zone_distribution(current_head_zones)}")
            print(f"  Conservation:     {current_conservation:.6f}")
            trace_data = model.trace()
            print(format_trace(step, trace_data))
            print("=== End Trace ===\n")

        # Decay the learning rate at the specified step.
        if step == config['t_lr_decay_step']:
            print('Decaying learning rate')
            for g in optimizer.param_groups:
                g['lr'] = config['t_lr_decayed']
    except StopIteration:
        # Handle the case where the training data iterator ends early.
        print("Training data iterator finished early.")
        break

# --- Save Model and Final Evaluation ---

# Create the output directory if it does not exist.
os.makedirs(config['t_out_path'].split('/')[0], exist_ok=True)

# Perform a final evaluation of the model on training and development datasets.
evaluation_losses = estimate_loss(200)
train_loss = evaluation_losses['train']
dev_loss = evaluation_losses['dev']

# Ensure unique model save path in case the file already exists.
modified_model_out_path = config['t_out_path']
save_tries = 0
while os.path.exists(modified_model_out_path):
    save_tries += 1
    model_out_name = os.path.splitext(config['t_out_path'])[0]
    modified_model_out_path = model_out_name + f"_{save_tries}" + ".pt"

# Save the model's state dictionary, optimizer state, and training metadata
# (including the runtime device / PyTorch / CUDA versions for reproducibility).
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': losses,
    'train_loss': train_loss,
    'dev_loss': dev_loss,
    'steps': len(losses),
    'device': config['device'],
    'pytorch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
}

if use_ternary:
    checkpoint['ternary_weights'] = True
    checkpoint['final_confidence'] = model.confidence
    checkpoint['final_block_zones'] = model.confidence_distribution()
    checkpoint['final_head_zones'] = model.head_zone_distribution()
    checkpoint['conservation_coeff'] = conservation_coeff

torch.save(checkpoint, modified_model_out_path)
print(f"Saved model to {modified_model_out_path}")
print(get_peak_memory_report(config['device']))

# Final ternary summary
if use_ternary:
    final_confidence = model.confidence
    final_zones = model.confidence_distribution()
    final_head_zones = model.head_zone_distribution()
    print(f"Final model confidence: {final_confidence:.4f}")
    print(f"Final block zones: {format_zone_distribution(final_zones)}")
    print(f"Final head zones:  {format_zone_distribution(final_head_zones)}")

print(f"Finished training. Train loss: {train_loss:.4f}, Dev loss: {dev_loss:.4f}")
