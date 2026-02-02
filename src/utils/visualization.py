"""Visualization utilities for plotting Active Learning results."""
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import pandas as pd


# Colors for each strategy (matching notebook)
COLORS = {
    'random': '#808080',
    'entropy': '#2E86AB',
    'least_confidence': '#A23B72',
    'margin': '#F18F01',
    'cp_size': '#C73E1D',
    'cp_v_shaped': '#8B0000',
    'combined': '#3A7D44',
    'combined_v_shaped': '#006400'
}

# Display labels for strategies
LABELS = {
    'random': 'Random',
    'entropy': 'Entropy',
    'least_confidence': 'Least Conf',
    'margin': 'Margin',
    'cp_size': 'CP Size',
    'cp_v_shaped': 'CP V-shaped',
    'combined': 'Combined',
    'combined_v_shaped': 'Comb V-shaped'
}


def compute_aulc(labeled_sizes, accuracies):
    """Compute normalized Area Under Learning Curve (AULC).
    
    Uses trapezoidal rule to compute the area, then normalizes by 
    the total number of samples to get an average accuracy metric.
    
    Args:
        labeled_sizes: List of number of labeled samples
        accuracies: List of accuracies at each size
        
    Returns:
        Normalized AULC (essentially average accuracy across the learning curve)
    """
    # Compute raw area using trapezoid (newer NumPy API)
    raw_aulc = np.trapezoid(accuracies, labeled_sizes)
    
    # Normalize by total samples to get interpretable metric (avg accuracy)
    max_samples = max(labeled_sizes) if len(labeled_sizes) > 0 else 1
    normalized_aulc = raw_aulc / max_samples
    
    return normalized_aulc


def compute_accuracy_gap(results_dict, baseline='random'):
    """Compute accuracy gap compared to baseline strategy.
    
    Args:
        results_dict: Dictionary mapping strategy names to results
        baseline: Baseline strategy name (default: 'random')
        
    Returns:
        Dictionary mapping strategy names to accuracy gaps (arrays)
    """
    if baseline not in results_dict:
        print(f"Warning: Baseline strategy '{baseline}' not found in results")
        return {}
    
    baseline_acc = np.array(results_dict[baseline]['accuracies'])
    
    gaps = {}
    for strategy_name, results in results_dict.items():
        if strategy_name == baseline:
            gaps[strategy_name] = np.zeros_like(baseline_acc)
        else:
            strategy_acc = np.array(results['accuracies'])
            gaps[strategy_name] = strategy_acc - baseline_acc
    
    return gaps


def plot_accuracy_vs_samples(results_dict, save_path=None, figsize=(14, 6), dataset=None, ylim=None):
    """Plot accuracy vs number of labeled samples for all strategies.
    
    Args:
        results_dict: Dictionary mapping strategy names to results
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        dataset: Dataset name for intelligent y-axis scaling (e.g., 'svhn')
        ylim: Manual y-axis limits as tuple (min, max). Overrides automatic scaling.
    """
    plt.figure(figsize=figsize)
    
    all_accuracies = []
    for strategy_name, results in results_dict.items():
        plt.plot(
            results['labeled_sizes'],
            results['accuracies'],
            label=LABELS.get(strategy_name, strategy_name),
            color=COLORS.get(strategy_name, None),
            marker='o',
            markersize=3,
            linewidth=1.5
        )
        all_accuracies.extend(results['accuracies'])
    
    # Y-axis scaling priority: manual > dataset-specific > auto-scale
    if ylim:
        # Manual override
        plt.ylim(ylim[0], ylim[1])
    elif dataset and dataset.lower() == 'svhn':
        # For SVHN, show 80-100% range for better visibility
        plt.ylim(80, 100)
    elif all_accuracies:
        # Auto-scale with padding for other datasets
        min_acc = min(all_accuracies)
        max_acc = max(all_accuracies)
        padding = (max_acc - min_acc) * 0.05
        plt.ylim(min_acc - padding, max_acc + padding)
    
    plt.xlabel('Samples Trained', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Accuracy vs Training Samples', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()
    

def plot_accuracy_gap(results_dict, baseline='random', save_path=None, figsize=(14, 6)):
    """Plot accuracy gap compared to baseline strategy.
    
    Args:
        results_dict: Dictionary mapping strategy names to results
        baseline: Baseline strategy name (default: 'random')
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    gaps = compute_accuracy_gap(results_dict, baseline)
    
    if not gaps:
        return
    
    plt.figure(figsize=figsize)
    
    for strategy_name, gap in gaps.items():
        if strategy_name == baseline:
            continue  # Skip baseline itself
            
        results = results_dict[strategy_name]
        plt.plot(
            results['labeled_sizes'],
            gap,
            label=LABELS.get(strategy_name, strategy_name),
            color=COLORS.get(strategy_name, None),
            marker='o',
            markersize=3,
            linewidth=1.5
        )
    
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, 
                label=f'{LABELS.get(baseline, baseline)} (Baseline)')
    
    plt.xlabel('Samples Trained', fontsize=12)
    plt.ylabel('Accuracy Gap (%)', fontsize=12)
    plt.title(f'Accuracy Gap vs {LABELS.get(baseline, baseline)}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def plot_cp_coverage(results_dict, target_coverage=0.9, save_path=None, figsize=(14, 6)):
    """Plot conformal prediction coverage over rounds.
    
    Args:
        results_dict: Dictionary mapping strategy names to results
        target_coverage: Target coverage level (default: 0.9 for alpha=0.1)
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    for strategy_name, results in results_dict.items():
        plt.plot(
            results['labeled_sizes'],
            results['cp_coverage'],
            label=LABELS.get(strategy_name, strategy_name),
            color=COLORS.get(strategy_name, None),
            marker='o',
            markersize=3,
            linewidth=1.5
        )
    
    # Target coverage line
    plt.axhline(y=target_coverage, color='red', linestyle='--', 
                linewidth=1.5, label=f'Target ({target_coverage:.0%})')
    
    plt.xlabel('Samples Trained', fontsize=12)
    plt.ylabel('Coverage', fontsize=12)
    plt.title('Conformal Prediction Coverage', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def plot_cp_set_size(results_dict, save_path=None, figsize=(14, 6)):
    """Plot average conformal prediction set size.
    
    Args:
        results_dict: Dictionary mapping strategy names to results
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    
    for strategy_name, results in results_dict.items():
        plt.plot(
            results['labeled_sizes'],
            results['cp_avg_set_size'],
            label=LABELS.get(strategy_name, strategy_name),
            color=COLORS.get(strategy_name, None),
            marker='o',
            markersize=3,
            linewidth=1.5
        )
    
    plt.xlabel('Samples Trained', fontsize=12)
    plt.ylabel('Avg Set Size', fontsize=12)
    plt.title('Average Prediction Set Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()


def plot_all_metrics(results_dict, output_dir=None, show=True, dataset=None, ylim=None):
    """Plot main Active Learning metrics (Accuracy, Gap, AULC).
    
    Args:
        results_dict: Dictionary mapping strategy names to results
        output_dir: Optional directory to save plots
        show: Whether to display plots
        dataset: Dataset name for intelligent y-axis scaling (e.g., 'svhn')
        ylim: Manual y-axis limits as tuple (min, max). Overrides automatic scaling.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. Accuracy vs Samples
    ax = axes[0]
    all_accuracies = []
    for strategy_name, results in results_dict.items():
        ax.plot(
            results['labeled_sizes'],
            results['accuracies'],
            label=LABELS.get(strategy_name, strategy_name),
            color=COLORS.get(strategy_name, None),
            marker='o',
            markersize=3,
            linewidth=1.5  # Thinner lines
        )
        all_accuracies.extend(results['accuracies'])
    
    # Y-axis scaling priority: manual > dataset-specific > auto-scale
    if ylim:
        # Manual override
        ax.set_ylim(ylim[0], ylim[1])
    elif dataset and dataset.lower() == 'svhn':
        # For SVHN, show 80-100% range for better visibility
        ax.set_ylim(80, 100)
    elif all_accuracies:
        # Auto-scale with padding for other datasets
        min_acc = min(all_accuracies)
        max_acc = max(all_accuracies)
        padding = (max_acc - min_acc) * 0.05
        ax.set_ylim(min_acc - padding, max_acc + padding)
    
    ax.set_xlabel('Samples Trained', fontsize=13)
    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title('Accuracy vs Training Samples', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 2. Accuracy Gap vs Random
    ax = axes[1]
    if 'random' in results_dict:
        gaps = compute_accuracy_gap(results_dict, 'random')
        for strategy_name, gap in gaps.items():
            if strategy_name == 'random':
                continue
            results = results_dict[strategy_name]
            ax.plot(
                results['labeled_sizes'],
                gap,
                label=LABELS.get(strategy_name, strategy_name),
                color=COLORS.get(strategy_name, None),
                marker='o',
                markersize=3,
                linewidth=1.5  # Thinner lines
            )
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Random (Baseline)')
    ax.set_xlabel('Samples Trained', fontsize=13)
    ax.set_ylabel('Accuracy Gap (%)', fontsize=13)
    ax.set_title('Accuracy Gap vs Random', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 3. AULC Comparison (Bar chart)
    ax = axes[2]
    strategy_names = []
    aulc_values = []
    colors = []
    
    # Sort by AULC descending
    sorted_items = sorted(results_dict.items(), 
                         key=lambda x: compute_aulc(x[1]['labeled_sizes'], x[1]['accuracies']),
                         reverse=True)
    
    for strategy_name, results in sorted_items:
        aulc = compute_aulc(results['labeled_sizes'], results['accuracies'])
        strategy_names.append(LABELS.get(strategy_name, strategy_name))
        aulc_values.append(aulc)
        colors.append(COLORS.get(strategy_name, '#808080'))
    
    bars = ax.barh(strategy_names, aulc_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('AULC (Normalized - Avg Accuracy %)', fontsize=13)
    ax.set_title('AULC Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars (show as percentage)
    for i, (bar, val) in enumerate(zip(bars, aulc_values)):
        ax.text(val, i, f'  {val:.2f}%', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'al_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved AL metrics plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_cp_metrics(results_dict, output_dir=None, show=True):
    """Plot Conformal Prediction metrics (Coverage, Set Size, Zero Sets).
    
    Args:
        results_dict: Dictionary mapping strategy names to results
        output_dir: Optional directory to save plots
        show: Whether to display plots
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 1. CP Coverage
    ax = axes[0]
    for strategy_name, results in results_dict.items():
        ax.plot(
            results['labeled_sizes'],
            results['cp_coverage'],
            label=LABELS.get(strategy_name, strategy_name),
            color=COLORS.get(strategy_name, None),
            marker='o',
            markersize=3,
            linewidth=1.5
        )
    ax.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='Target (90%)')
    ax.set_xlabel('Samples Trained', fontsize=13)
    ax.set_ylabel('Coverage', fontsize=13)
    ax.set_title('CP Coverage', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 2. Avg Set Size
    ax = axes[1]
    for strategy_name, results in results_dict.items():
        ax.plot(
            results['labeled_sizes'],
            results['cp_avg_set_size'],
            label=LABELS.get(strategy_name, strategy_name),
            color=COLORS.get(strategy_name, None),
            marker='o',
            markersize=3,
            linewidth=1.5
        )
    ax.set_xlabel('Samples Trained', fontsize=13)
    ax.set_ylabel('Avg Set Size', fontsize=13)
    ax.set_title('Average Prediction Set Size', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 3. Zero Sets (Overconfident Errors)
    ax = axes[2]
    for strategy_name, results in results_dict.items():
        ax.plot(
            results['labeled_sizes'],
            results['cp_zero_sets'],
            label=LABELS.get(strategy_name, strategy_name),
            color=COLORS.get(strategy_name, None),
            marker='o',
            markersize=3,
            linewidth=1.5
        )
    ax.set_xlabel('Samples Trained', fontsize=13)
    ax.set_ylabel('Number of Zero Sets', fontsize=13)
    ax.set_title('Overconfident Errors (Empty Sets)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'cp_metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved CP metrics plot to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def load_results(results_dir):
    """Load all results from a directory.
    
    Args:
        results_dir: Directory containing results.pt files
        
    Returns:
        Dictionary mapping strategy names to results
    """
    results_dict = {}
    
    # Walk through directory to find results.pt files
    for root, dirs, files in os.walk(results_dir):
        if 'results.pt' in files:
            results_path = os.path.join(root, 'results.pt')
            results = torch.load(results_path)
            strategy_name = results['strategy']
            results_dict[strategy_name] = results
            print(f"Loaded results for: {strategy_name}")
    
    return results_dict


def export_accuracy_table(results_dict, output_path='accuracy_table.csv'):
    """Export accuracy vs samples table to CSV.
    
    Args:
        results_dict: Dictionary mapping strategy names to results
        output_path: Path to save CSV file (default: 'accuracy_table.csv')
        
    Returns:
        DataFrame with accuracy table
    """
    # Find all unique sample sizes
    all_samples = set()
    for results in results_dict.values():
        all_samples.update(results['labeled_sizes'])
    
    sample_sizes = sorted(all_samples)
    
    # Create DataFrame
    data = {'Samples': sample_sizes}
    
    for strategy_name in sorted(results_dict.keys()):
        results = results_dict[strategy_name]
        label = LABELS.get(strategy_name, strategy_name)
        
        # Map accuracies to sample sizes
        accuracy_map = dict(zip(results['labeled_sizes'], results['accuracies']))
        data[label] = [accuracy_map.get(s, None) for s in sample_sizes]
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"\n📊 Exported accuracy table to: {output_path}")
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    
    return df


def print_summary_table(results_dict, show_cp_metrics=False):
    """Print a summary table of final results.
    
    Args:
        results_dict: Dictionary mapping strategy names to results
        show_cp_metrics: Whether to show CP metrics (default: False)
    """
    print("\n" + "="*75)
    print("ACTIVE LEARNING RESULTS")
    print("="*75)
    print(f"{'Strategy':<20} {'Final Acc':<12} {'AULC (Norm)':<15}")
    print("-"*75)
    
    for strategy_name in sorted(results_dict.keys()):
        results = results_dict[strategy_name]
        final_acc = results['accuracies'][-1]
        aulc = compute_aulc(results['labeled_sizes'], results['accuracies'])
        
        label = LABELS.get(strategy_name, strategy_name)
        print(f"{label:<20} {final_acc:>10.2f}%  {aulc:>13.2f}%")
    
    print("="*75)
    
    # Print accuracy gaps vs random if available
    if 'random' in results_dict:
        print("\n" + "="*50)
        print("Accuracy Gap vs Random (Final Round)")
        print("="*50)
        baseline_acc = results_dict['random']['accuracies'][-1]
        
        # Sort by gap (descending)
        gaps_sorted = []
        for strategy_name in results_dict.keys():
            if strategy_name == 'random':
                continue
            results = results_dict[strategy_name]
            gap = results['accuracies'][-1] - baseline_acc
            gaps_sorted.append((strategy_name, gap))
        
        gaps_sorted.sort(key=lambda x: x[1], reverse=True)
        
        for strategy_name, gap in gaps_sorted:
            label = LABELS.get(strategy_name, strategy_name)
            sign = '+' if gap >= 0 else ''
            print(f"{label:<20} {sign}{gap:>6.2f}%")
        print("="*50)
    
    # Optionally show CP metrics
    if show_cp_metrics:
        print("\n" + "="*70)
        print("CONFORMAL PREDICTION METRICS")
        print("="*70)
        print(f"{'Strategy':<20} {'Coverage':<12} {'Avg Set':<12} {'Zero Sets':<12}")
        print("-"*70)
        
        for strategy_name in sorted(results_dict.keys()):
            results = results_dict[strategy_name]
            final_cov = results['cp_coverage'][-1]
            final_set = results['cp_avg_set_size'][-1]
            final_zero = results['cp_zero_sets'][-1]
            
            label = LABELS.get(strategy_name, strategy_name)
            print(f"{label:<20} {final_cov:>10.3f}  {final_set:>10.2f}  {final_zero:>11d}")
        
        print("="*70)
    
    print()
