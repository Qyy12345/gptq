#!/usr/bin/env python3
"""
Plot layer-wise quantization errors from GPTQ log files.
Automatically groups logs by bitwidth and creates comparison plots.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Configuration: List of log files to ignore
IGNORE_LOGS = [
    'r1_allcol_opt125m_c4.log',
    '4_opt125m_wiki.log',
]


def parse_log_file(log_path):
    """
    Parse a GPTQ log file and extract layer names, error values, and perplexity.

    Args:
        log_path: Path to the log file

    Returns:
        tuple: (layer_names, error_values, perplexity)
            - layer_names: List of layer identifiers
            - error_values: List of corresponding error values
            - perplexity: Perplexity value from last line (or None if not found)
    """
    layer_names = []
    error_values = []
    perplexity = None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    current_layer = None

    for i, line in enumerate(lines):
        line = line.strip()

        # Match layer name (e.g., "0 self_attn.k_proj")
        layer_match = re.match(r'^(\d+\s+self_attn\.(?:k_proj|v_proj|q_proj|out_proj)|\d+\s+fc[12])$', line)
        if layer_match:
            current_layer = layer_match.group(1)

        # Match error line (e.g., "error 12432.7109375")
        error_match = re.match(r'^error\s+([\d.]+)', line)
        if error_match and current_layer:
            error_value = float(error_match.group(1))
            layer_names.append(current_layer)
            error_values.append(error_value)
            current_layer = None

        # Try to read perplexity from the last non-empty line
        if i == len(lines) - 1 or (i == len(lines) - 2 and not lines[-1].strip()):
            try:
                perplexity = float(line)
            except (ValueError, IndexError):
                pass

    return layer_names, error_values, perplexity


def get_bitwidth_and_type(log_path):
    """
    Extract bitwidth, model, config, and dataset from log filename.

    Naming convention: {bitwidth}_{config}_{model}_{dataset}.log
    - Position 1: bitwidth (2/3/4)
    - Position 2: config (empty/sym/r2/r2nosym), empty means baseline
    - Position 3: model name
    - Position 4: dataset name

    Examples:
    - 2_r2_opt125m_c4.log → bitwidth=2, config=r2, model=opt125m
    - 2_opt125m_c4.log → bitwidth=2, config="", model=opt125m (baseline)
    - 4_opt125m_c4.log → bitwidth=4, config="", model=opt125m (baseline)
    - r2_opt125m_c4.log → bitwidth=4, config=r2, model=opt125m

    Args:
        log_path: Path to the log file

    Returns:
        tuple: (bitwidth, model, config, display_name)
            - bitwidth: Integer bitwidth (2, 3, 4, etc.)
            - model: Model name (e.g., 'opt125m', 'llama-7b')
            - config: Configuration name (e.g., '', 'r2', 'sym', 'nosym')
            - display_name: Clean name for display (e.g., 'baseline', 'r2', 'r2nosym')
    """
    filename = Path(log_path).stem

    # Try to match pattern: {bitwidth}_{config}_{model}_{dataset}
    # Known configs: r1/r2/r3/r4/r5, (r1/r2/r3/r4/r5)nosym, (r1/r2/r3/r4/r5)noslice, addsym, noslice, allcol
    match = re.match(r'^(\d+)_(r[12345](?:nosym|noslice)?|addsym|noslice|allcol)_(.+)_(c4|wiki|owt|ptb)$', filename)
    if match:
        bitwidth = int(match.group(1))
        config = match.group(2)
        model = match.group(3)
        dataset = match.group(4)

        # Convert 'sym' to 'addsym' for display
        if config == 'sym':
            config = 'addsym'
            display_name = 'addsym'
        else:
            display_name = config

        return bitwidth, model, config, display_name

    # Try to match pattern: {bitwidth}_{model}_{dataset} (baseline - config position is empty)
    match = re.match(r'^(\d+)_(opt125m|opt[0-9.]+b|llama[0-9]+b|gpt[0-9]+|llama_[0-9]+b)_(c4|wiki|owt|ptb)$', filename)
    if match:
        bitwidth = int(match.group(1))
        config = ''
        model = match.group(2)
        dataset = match.group(3)

        display_name = 'baseline'
        return bitwidth, model, config, display_name

    # Try to match pattern: {config}_{model}_{dataset} (no bitwidth prefix, default to 4)
    match = re.match(r'^(r[12345](?:nosym|noslice)?|addsym|noslice|allcol)_(.+)_(c4|wiki|owt|ptb)$', filename)
    if match:
        bitwidth = 4
        config = match.group(1)
        model = match.group(2)
        dataset = match.group(3)

        display_name = config
        return bitwidth, model, config, display_name

    # Try to match pattern: {model}_{dataset} (no bitwidth, no config, default to 4, baseline)
    match = re.match(r'^(opt125m|opt[0-9.]+b|llama[0-9]+b|gpt[0-9]+|llama_[0-9]+b)_(c4|wiki|owt|ptb)$', filename)
    if match:
        bitwidth = 4
        config = ''
        model = match.group(1)
        dataset = match.group(2)

        display_name = 'baseline'
        return bitwidth, model, config, display_name

    # DEBUG: print if we reach fallback
    import sys
    print(f"DEBUG: Using fallback for {filename}", file=sys.stderr)

    # Fallback: try to parse any {model}_{dataset} pattern
    match = re.match(r'^(.+)_([a-z]+[0-9]*)$', filename)
    if match:
        bitwidth = 4
        config = ''
        model = match.group(1)
        dataset = match.group(2)

        display_name = 'baseline'
        return bitwidth, model, config, display_name

    # Ultimate fallback
    bitwidth = 4
    model = filename
    config = ''
    display_name = 'baseline'

    return bitwidth, model, config, display_name


def get_baseline_for_group(logs_by_group, group_key):
    """
    Determine the baseline log file for a given (bitwidth, model) group.
    The baseline should be the one named "baseline" (no config modifiers).
    Prefer baselines without extra suffixes (like 'wiki').

    Args:
        logs_by_group: Dictionary mapping (bitwidth, model) to list of (log_path, display_name)
        group_key: Tuple of (bitwidth, model)

    Returns:
        str or None: Path to baseline log file, or None if not found
    """
    if group_key not in logs_by_group:
        return None

    logs = logs_by_group[group_key]

    # Look for file named "baseline" (this has highest priority)
    # Prefer baselines without extra suffixes like 'wiki'
    primary_baseline = None
    for log_path, display_name in logs:
        if display_name == 'baseline':
            # Check if this is a primary baseline (no wiki, etc.)
            if 'wiki' not in Path(log_path).stem.lower():
                return log_path
            elif primary_baseline is None:
                primary_baseline = log_path

    # If no primary baseline found, use the wiki baseline
    if primary_baseline:
        return primary_baseline

    # If no "baseline" found, try to find files without common modifiers
    modifiers = ['r1', 'r1nosym', 'r1noslice', 'r2', 'r2nosym', 'r2noslice', 'r3', 'r3nosym', 'r3noslice', 'r4', 'r4nosym', 'r4noslice', 'r5', 'r5nosym', 'r5noslice', 'addsym', 'noslice', 'allcol']
    for log_path, display_name in logs:
        # Check if this name doesn't contain any modifiers
        if not any(mod in display_name for mod in modifiers):
            return log_path

    # Fallback: use the first one
    return logs[0][0] if logs else None


def plot_difference_for_group(logs_dict, bitwidth, model, baseline_path, output_dir):
    """
    Create a difference plot for a specific (bitwidth, model) group.

    Args:
        logs_dict: Dictionary mapping display names to (layer_names, error_values, original_path, perplexity)
        bitwidth: The bitwidth being plotted
        model: The model being plotted
        baseline_path: Path to the baseline log file
        output_dir: Directory to save output figures
    """
    print(f"\n{'='*60}")
    print(f"Generating plot for bitwidth={bitwidth}, model={model}")
    print(f"Baseline: {Path(baseline_path).name}")
    print(f"{'='*60}")

    # Find baseline data
    baseline_data = None
    baseline_perplexity = None
    for display_name, data in logs_dict.items():
        if data[2] == baseline_path:  # data[2] is the original_path
            baseline_data = (display_name, data[0], data[1])
            baseline_perplexity = data[3]  # data[3] is perplexity
            break

    if not baseline_data:
        print(f"Error: Baseline data not found for {baseline_path}")
        print(f"Available files in logs_dict:")
        for display_name, data in logs_dict.items():
            print(f"  {display_name}: {data[2]}")
        return

    baseline_name, baseline_layers, baseline_errors = baseline_data
    baseline_dict = dict(zip(baseline_layers, baseline_errors))

    # Find common layers across all logs in this group
    common_layers = set(baseline_layers)
    for display_name, (layer_names, _, _, _, _) in logs_dict.items():
        common_layers &= set(layer_names)
    common_layers = sorted(list(common_layers))

    print(f"Found {len(common_layers)} common layers across {len(logs_dict)} log(s)")

    if len(common_layers) == 0 or len(logs_dict) < 2:
        print("Skipping: Need at least 2 logs with common layers")
        return

    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    x_positions = np.arange(len(common_layers))

    colors = ['#A23B72', '#F18F01', '#06A77D', '#2E86AB', '#8B4513']

    # Collect perplexity info for display
    ppl_info_lines = []
    ppl_info_lines.append(f"Baseline: {baseline_name}")
    if baseline_perplexity:
        ppl_info_lines.append(f"Baseline PPL: {baseline_perplexity:.2f}")

    # Plot each log file compared to baseline
    plot_idx = 0
    for display_name, (layer_names, error_values, _, perplexity, _) in logs_dict.items():
        if display_name == baseline_name:
            continue  # Skip baseline itself

        error_dict = dict(zip(layer_names, error_values))
        baseline_vals = [baseline_dict[layer] for layer in common_layers]
        current_vals = [error_dict[layer] for layer in common_layers]
        diff = [(c - b) / b * 100 for c, b in zip(current_vals, baseline_vals)]

        color = colors[plot_idx % len(colors)]
        ax.plot(x_positions, diff, marker='o', linewidth=2.5,
               markersize=8, label=f'{display_name}',
               color=color)

        # Collect perplexity info
        ppl_str = f"PPL: {perplexity:.2f}" if perplexity else "PPL: N/A"
        ppl_info_lines.append(f"{display_name}: {ppl_str}")

        # Add value labels
        for j, (x, y) in enumerate(zip(x_positions, diff)):
            ax.annotate(f'{y:+.1f}%', (x, y), textcoords="offset points",
                       xytext=(0, 10 if y >= 0 else -20), ha='center',
                       fontsize=7, rotation=0, color=color)

        plot_idx += 1

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Difference from Baseline (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Quantization Error Comparison ({bitwidth}-bit, {model})',
                fontsize=16, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(common_layers, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add perplexity info box in upper left corner
    if len(ppl_info_lines) > 1:
        ppl_text = '\n'.join(ppl_info_lines)
        ax.text(0.02, 0.98, ppl_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save the figure
    output_file = Path(output_dir) / f'layer_errors_{bitwidth}_{model}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_file}")
    plt.close()


def should_ignore_log(log_path):
    """
    Check if a log file should be ignored.

    Args:
        log_path: Path to the log file

    Returns:
        bool: True if should ignore, False otherwise
    """
    return log_path.name in IGNORE_LOGS


def plot_all_bitwidths(log_dir='logs', output_dir='logs'):
    """
    Automatically detect and plot all (bitwidth, model) groups.

    Args:
        log_dir: Directory containing log files
        output_dir: Directory to save output figures
    """
    # Find all log files and group by (bitwidth, model)
    logs_by_group = defaultdict(list)  # (bitwidth, model) -> [(log_path, display_name)]
    log_data = {}  # (bitwidth, model, display_name) -> (layer_names, error_values, original_path, perplexity)

    log_files = list(Path(log_dir).glob('*.log'))

    print(f"Found {len(log_files)} log files in {log_dir}/")

    for log_path in log_files:
        # Skip ignored log files
        if should_ignore_log(log_path):
            print(f"  {log_path.name}: SKIPPED (in ignore list)")
            continue

        # Parse the log file
        layer_names, error_values, perplexity = parse_log_file(log_path)

        # Get bitwidth, model, config, and display name
        bitwidth, model, config, display_name = get_bitwidth_and_type(log_path)

        group_key = (bitwidth, model)
        # Create a unique key that includes the filename to avoid collisions
        # When multiple files have the same display_name (e.g., both baselines)
        unique_key = f"{display_name}_{Path(log_path).stem}"
        logs_by_group[group_key].append((str(log_path), display_name))
        log_data[(bitwidth, model, unique_key)] = (layer_names, error_values, str(log_path), perplexity, display_name)

        ppl_str = f", ppl={perplexity:.2f}" if perplexity else ""
        print(f"  {log_path.name}: bitwidth={bitwidth}, model={model}, config={config if config else 'baseline'}, {len(layer_names)} layers{ppl_str}")

    if not logs_by_group:
        print("Error: No log files found!")
        return

    # Generate plot for each (bitwidth, model) group
    for group_key in sorted(logs_by_group.keys()):
        bitwidth, model = group_key
        baseline_path = get_baseline_for_group(logs_by_group, group_key)

        if not baseline_path:
            print(f"\nWarning: No baseline found for bitwidth={bitwidth}, model={model}, skipping...")
            continue

        # Create logs dictionary for this group
        logs_dict = {}
        for log_path, display_name in logs_by_group[group_key]:
            # Create unique key based on file characteristics
            if display_name == 'baseline':
                # Add suffix to differentiate multiple baselines
                if 'wiki' in Path(log_path).stem.lower():
                    key = 'baseline_wiki'
                else:
                    key = 'baseline'
            else:
                key = display_name

            # Find the correct unique_key in log_data
            unique_key = f"{display_name}_{Path(log_path).stem}"

            # Store in logs_dict (store display_name separately for later use)
            logs_dict[key] = log_data[(bitwidth, model, unique_key)]

        # Generate the plot
        plot_difference_for_group(logs_dict, bitwidth, model, baseline_path, output_dir)

    print(f"\n{'='*60}")
    print("All plots generated successfully!")
    print(f"{'='*60}")

    # Display one of the plots (the last one generated)
    if len(logs_by_group) > 0:
        last_group = sorted(logs_by_group.keys())[-1]
        last_bitwidth, last_model = last_group
        display_file = Path(output_dir) / f'layer_errors_{last_bitwidth}_{last_model}.png'
        print(f"\nDisplaying plot for bitwidth={last_bitwidth}, model={last_model}...")


def main():
    """Main function to run the visualization."""
    # You can customize these directories if needed
    # Note: script assumes it's run from the parent directory of visual/
    import sys
    if Path('visual').exists():
        # Running from parent directory
        log_directory = 'logs'
        output_directory = 'logs'
    else:
        # Running from visual/ directory
        log_directory = '../logs'
        output_directory = '../logs'

    plot_all_bitwidths(log_directory, output_directory)


if __name__ == '__main__':
    main()
