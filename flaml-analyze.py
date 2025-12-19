#!/usr/bin/env python3
"""
flaml-analyze.py :: extract best configurations from FLAML optimization logs.

This script:
1. Parses FLAML log files (JSON lines format)
2. Extracts the top-n best configurations for each learner
3. Analyzes optimization progress over time
4. Generates plots showing performance and convergence
5. Saves best configs in a format ready for model training
6. Saves configurations for warm start (top N overall + top N per method)
7. Filters out unknown methods from plots and data display

https://github.com/filipsPL/flaml-log-analyze/
DOI 10.5281/zenodo.17987939

PATCHED VERSION: Fixed type safety issues identified by Pyre
Signed version
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, DefaultDict, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans


def parse_flaml_log(log_file: Path) -> List[Dict]:
    """Parse FLAML log file (JSON lines format)."""
    records = []
    with open(log_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {line[:100]}... Error: {e}")
    return records


def filter_unknown_methods(records: List[Dict]) -> List[Dict]:
    """Filter out records with unknown or missing learner/method."""
    filtered = [r for r in records if r.get('learner') and r.get('learner') != 'unknown']
    removed = len(records) - len(filtered)
    if removed > 0:
        print(f"Filtered out {removed} records with unknown/missing method")
    return filtered


def extract_best_configs(records: List[Dict], n_best: int = 1) -> Dict[str, List[Dict]]:
    """
    Extract top-n best configurations for each learner.
    
    Args:
        records: List of FLAML log records
        n_best: Number of best configurations to extract per learner
        
    Returns:
        Dictionary mapping learner name to list of best configs
    """
    # Group records by learner
    learner_records: DefaultDict[str, List[Dict]] = defaultdict(list)
    for record in records:
        learner = record.get('learner')
        if learner:
            learner_records[learner].append(record)
    
    # Extract top-n for each learner
    best_configs: Dict[str, List[Dict]] = {}
    for learner, recs in learner_records.items():
        # Sort by validation_loss (lower is better)
        sorted_recs = sorted(recs, key=lambda x: x.get('validation_loss', float('inf')))
        best_configs[learner] = sorted_recs[:n_best]
    
    return best_configs


def analyze_optimization_progress(records: List[Dict]) -> Dict[str, Any]:
    """
    Analyze optimization progress across all trials.
    
    Returns:
        Dictionary with analysis results
    """
    # FIX: Properly type the learners defaultdict with explicit structure
    learner_stats: DefaultDict[str, Dict[str, Any]] = defaultdict(
        lambda: {'count': 0, 'best_loss': float('inf')}
    )
    
    analysis: Dict[str, Any] = {
        'total_trials': len(records),
        'learners': learner_stats,
        'cumulative_best': [],
        'wall_clock_times': [],
        'validation_losses': [],
        'train_losses': [],
        'trial_times': [],
        'learner_sequence': [],
        'best_overall': None
    }
    
    current_best = float('inf')
    
    for record in records:
        learner = record.get('learner', 'unknown')
        val_loss = record.get('validation_loss', float('inf'))
        train_loss = record.get('logged_metric', {}).get('train_loss', float('inf'))
        trial_time = record.get('trial_time', 0)
        wall_time = record.get('wall_clock_time', 0)
        
        # FIX: Explicitly ensure learner entry exists before updating
        if learner not in analysis['learners']:
            analysis['learners'][learner] = {'count': 0, 'best_loss': float('inf')}
        
        # Update learner stats
        analysis['learners'][learner]['count'] += 1
        if val_loss < analysis['learners'][learner]['best_loss']:
            analysis['learners'][learner]['best_loss'] = val_loss
        
        # Track cumulative best
        if val_loss < current_best:
            current_best = val_loss
        analysis['cumulative_best'].append(current_best)
        
        # Track other metrics
        analysis['wall_clock_times'].append(wall_time)
        analysis['validation_losses'].append(val_loss)
        analysis['train_losses'].append(train_loss)
        analysis['trial_times'].append(trial_time)
        analysis['learner_sequence'].append(learner)
        
        # Track best overall
        if val_loss == current_best:
            analysis['best_overall'] = record
    
    return analysis


def plot_optimization_progress(analysis: Dict[str, Any], output_file: Path):
    """
    Create comprehensive visualization of optimization progress.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Cumulative Best Loss over Trials
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(len(analysis['cumulative_best'])), 
             analysis['cumulative_best'], 
             linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Trial Number', fontsize=11)
    ax1.set_ylabel('Best Validation Loss', fontsize=11)
    ax1.set_title('Optimization Progress: Cumulative Best', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. All Validation Losses (scatter by learner)
    ax2 = fig.add_subplot(gs[0, 1])
    learners_unique = list(set(analysis['learner_sequence']))
    colors = plt.cm.tab10(np.linspace(0, 1, len(learners_unique)))
    color_map = dict(zip(learners_unique, colors))
    
    for learner in learners_unique:
        indices = [i for i, l in enumerate(analysis['learner_sequence']) if l == learner]
        losses = [analysis['validation_losses'][i] for i in indices]
        ax2.scatter(indices, losses, label=learner, alpha=0.6, s=30, color=color_map[learner])
    
    ax2.set_xlabel('Trial Number', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('All Trials by Learner', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Train vs Validation Loss
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(analysis['train_losses'], analysis['validation_losses'], 
                alpha=0.5, s=30, color='#A23B72')
    max_train = max(analysis['train_losses']) if analysis['train_losses'] else 1
    ax3.plot([0, max_train], 
             [0, max_train], 
             'k--', alpha=0.3, label='Perfect fit')
    ax3.set_xlabel('Train Loss', fontsize=11)
    ax3.set_ylabel('Validation Loss', fontsize=11)
    ax3.set_title('Train vs Validation Loss', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Trial Times
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(range(len(analysis['trial_times'])), 
             analysis['trial_times'], 
             linewidth=1, alpha=0.7, color='#F18F01')
    ax4.set_xlabel('Trial Number', fontsize=11)
    ax4.set_ylabel('Trial Time (seconds)', fontsize=11)
    ax4.set_title('Computational Cost per Trial', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Learner Performance Distribution (Box plot)
    ax5 = fig.add_subplot(gs[2, 0])
    learner_losses: DefaultDict[str, List[float]] = defaultdict(list)
    for learner, loss in zip(analysis['learner_sequence'], analysis['validation_losses']):
        learner_losses[learner].append(loss)
    
    # Use 'labels' parameter for matplotlib < 3.5 compatibility (tick_labels was added in 3.5)
    bp = ax5.boxplot([learner_losses[l] for l in learners_unique], 
                      labels=learners_unique,
                      patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax5.set_xlabel('Learner', fontsize=11)
    ax5.set_ylabel('Validation Loss', fontsize=11)
    ax5.set_title('Performance Distribution by Learner', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 6. Learner Trial Counts
    ax6 = fig.add_subplot(gs[2, 1])
    learner_counts = [analysis['learners'][l]['count'] for l in learners_unique]
    bars = ax6.bar(learners_unique, learner_counts, color=colors, alpha=0.7)
    ax6.set_xlabel('Learner', fontsize=11)
    ax6.set_ylabel('Number of Trials', fontsize=11)
    ax6.set_title('Trial Distribution by Learner', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('FLAML Optimization Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()


def plot_search_space_2d(records: List[Dict], absolute_configs: Dict[str, List[Dict]], 
                         representative_configs: Dict[str, List[Dict]], output_file: Path):
    """
    Create 2D PCA projection visualization of the search space.
    Shows all trials colored by performance with overlays for selected configs.
    
    Args:
        records: All FLAML trial records
        absolute_configs: Dict of absolute top-N configs per learner
        representative_configs: Dict of representative configs per learner
        output_file: Path to save the plot
    """
    from sklearn.decomposition import PCA
    
    print("\nGenerating search space visualization...")
    
    # Group records by learner
    learner_records: DefaultDict[str, List[Dict]] = defaultdict(list)
    for record in records:
        learner = record.get('learner')
        if learner:
            learner_records[learner].append(record)
    
    # Calculate number of subplots needed
    n_learners = len(learner_records)
    if n_learners == 0:
        print("  No learners found, skipping visualization")
        return
    
    # Create figure with subplots for each learner
    n_cols = min(3, n_learners)
    n_rows = (n_learners + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_learners == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten() if n_rows > 1 else np.array(axes)
    
    for idx, (learner, recs) in enumerate(sorted(learner_records.items())):
        ax = axes[idx]
        
        if len(recs) < 3:
            ax.text(0.5, 0.5, f'{learner}\nToo few trials for visualization',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{learner} ({len(recs)} trials)')
            continue
        
        # Encode configurations
        all_params = set()
        for rec in recs:
            all_params.update(rec.get('config', {}).keys())
        all_params.discard('FLAML_sample_size')
        param_names = sorted(all_params)
        
        if len(param_names) == 0:
            ax.text(0.5, 0.5, f'{learner}\nNo parameters found',
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Identify categorical parameters
        categorical_params = set()
        for param in param_names:
            for rec in recs:
                value = rec.get('config', {}).get(param)
                if value is not None and not isinstance(value, (int, float)):
                    categorical_params.add(param)
                    break
        
        # Create encoders for categorical parameters
        categorical_encoders: Dict[str, LabelEncoder] = {}
        for param in categorical_params:
            values = [rec.get('config', {}).get(param) for rec in recs 
                     if rec.get('config', {}).get(param) is not None]
            if values:
                encoder = LabelEncoder()
                encoder.fit(values)
                categorical_encoders[param] = encoder
        
        # Encode all configurations
        config_vectors = []
        validation_losses = []
        for rec in recs:
            vector = encode_config_to_vector(rec.get('config', {}), param_names, 
                                            categorical_encoders)
            config_vectors.append(vector)
            validation_losses.append(rec.get('validation_loss', float('inf')))
        
        X = np.array(config_vectors)
        validation_losses_array = np.array(validation_losses)
        
        # Handle missing values
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            valid_mask = col != -999
            if valid_mask.any():
                col_mean = col[valid_mask].mean()
                col[~valid_mask] = col_mean
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for 2D projection
        n_components = min(2, X_scaled.shape[1])
        if n_components < 2:
            ax.text(0.5, 0.5, f'{learner}\nInsufficient dimensions for PCA',
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(X_scaled)
        
        # Plot all trials colored by performance
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1],
                           c=validation_losses_array, cmap='RdYlGn_r',
                           s=50, alpha=0.5, edgecolors='none')
        
        # Get indices of selected configs
        absolute_indices = []
        for abs_cfg in absolute_configs.get(learner, []):
            abs_metric = abs_cfg.get('metric')
            for i, rec in enumerate(recs):
                if abs(rec.get('validation_loss', float('inf')) - abs_metric) < 1e-6:
                    absolute_indices.append(i)
                    break
        
        representative_indices = []
        for rep_cfg in representative_configs.get(learner, []):
            rep_metric = rep_cfg.get('metric')
            for i, rec in enumerate(recs):
                if abs(rec.get('validation_loss', float('inf')) - rep_metric) < 1e-6:
                    representative_indices.append(i)
                    break
        
        # Overlay absolute configs (red stars)
        if absolute_indices:
            ax.scatter(coords_2d[absolute_indices, 0],
                      coords_2d[absolute_indices, 1],
                      marker='*', s=400, c='red', 
                      edgecolors='black', linewidths=0.5,
                      label='Absolute', zorder=5)
        
        # Overlay representative configs (blue diamonds)
        if representative_indices:
            ax.scatter(coords_2d[representative_indices, 0],
                      coords_2d[representative_indices, 1],
                      marker='D', s=200, c='blue',
                      edgecolors='black', linewidths=0.5,
                      label='Representative', zorder=5)
        
        # Formatting
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=10)
        ax.set_title(f'{learner} ({len(recs)} trials)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Validation Loss')
    
    # Hide empty subplots
    for idx in range(len(learner_records), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Search Space Exploration (2D PCA Projection)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Search space plot saved to: {output_file}")
    plt.close()


def generate_text_summary(analysis: Dict[str, Any], best_configs: Dict[str, List[Dict]], 
                         output_file: Path):
    """Generate detailed text summary of optimization results."""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FLAML OPTIMIZATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total trials: {analysis['total_trials']}\n")
        
        if analysis['validation_losses']:
            f.write(f"Best validation loss: {min(analysis['validation_losses']):.6f}\n")
            f.write(f"Worst validation loss: {max(analysis['validation_losses']):.6f}\n")
            f.write(f"Mean validation loss: {np.mean(analysis['validation_losses']):.6f}\n")
            f.write(f"Std validation loss: {np.std(analysis['validation_losses']):.6f}\n")
        
        if analysis['wall_clock_times']:
            total_time = max(analysis['wall_clock_times'])
            f.write(f"Total wall clock time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        
        if analysis['trial_times']:
            mean_trial_time = np.mean(analysis['trial_times'])
            f.write(f"Mean trial time: {mean_trial_time:.2f} seconds\n")
        
        f.write("\n")
        
        # Learner statistics
        f.write("LEARNER STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Learner':<15} {'Trials':<10} {'Best Loss':<15} {'Mean Loss':<15}\n")
        f.write("-" * 80 + "\n")
        
        for learner in sorted(analysis['learners'].keys()):
            count = analysis['learners'][learner]['count']
            best_loss = analysis['learners'][learner]['best_loss']
            learner_losses = [loss for l, loss in zip(analysis['learner_sequence'], 
                                                       analysis['validation_losses']) 
                              if l == learner]
            mean_loss = np.mean(learner_losses) if learner_losses else float('inf')
            f.write(f"{learner:<15} {count:<10} {best_loss:<15.6f} {mean_loss:<15.6f}\n")
        
        f.write("\n")
        
        # Best Overall Configuration
        f.write("BEST OVERALL CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        if analysis['best_overall']:
            best = analysis['best_overall']
            f.write(f"Learner: {best.get('learner')}\n")
            f.write(f"Validation Loss: {best.get('validation_loss'):.6f}\n")
            train_loss = best.get('logged_metric', {}).get('train_loss', 'N/A')
            if train_loss != 'N/A':
                f.write(f"Train Loss: {train_loss}\n")
            trial_time = best.get('trial_time', 'N/A')
            if trial_time != 'N/A':
                f.write(f"Trial Time: {trial_time:.2f} seconds\n")
            f.write(f"Configuration:\n")
            config = best.get('config', {})
            for key, value in sorted(config.items()):
                if key != 'FLAML_sample_size':
                    f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # Top Configurations per Learner
        f.write("TOP CONFIGURATIONS BY LEARNER\n")
        f.write("=" * 80 + "\n\n")
        
        for learner in sorted(best_configs.keys()):
            f.write(f"Learner: {learner.upper()}\n")
            f.write("-" * 80 + "\n")
            
            for idx, record in enumerate(best_configs[learner], 1):
                f.write(f"\nRank #{idx}:\n")
                f.write(f"  Validation Loss: {record.get('validation_loss'):.6f}\n")
                train_loss = record.get('logged_metric', {}).get('train_loss', 'N/A')
                if train_loss != 'N/A':
                    f.write(f"  Train Loss: {train_loss}\n")
                trial_time = record.get('trial_time', 'N/A')
                if trial_time != 'N/A':
                    f.write(f"  Trial Time: {trial_time:.2f} seconds\n")
                f.write(f"  Configuration:\n")
                config = record.get('config', {})
                for key, value in sorted(config.items()):
                    if key != 'FLAML_sample_size':
                        f.write(f"    {key}: {value}\n")
            
            f.write("\n")
    
    print(f"Summary saved to: {output_file}")


def encode_config_to_vector(config: Dict, param_names: List[str], 
                            categorical_encoders: Dict[str, LabelEncoder]) -> np.ndarray:
    """
    Encode a configuration dictionary to a numerical vector.
    
    Args:
        config: Configuration dictionary
        param_names: Ordered list of parameter names
        categorical_encoders: Dictionary of LabelEncoders for categorical params
    
    Returns:
        Numpy array representing the configuration
    """
    vector = []
    for param in param_names:
        value = config.get(param)
        
        if value is None:
            # Missing value - use a sentinel (e.g., -999)
            vector.append(-999)
        elif param in categorical_encoders:
            # Categorical parameter - use encoded value
            try:
                encoded = categorical_encoders[param].transform([value])[0]
                vector.append(encoded)
            except ValueError:
                # Unknown category - use -1
                vector.append(-1)
        else:
            # Numerical parameter
            vector.append(float(value))
    
    return np.array(vector)


def select_diverse_configs(records: List[Dict], n_select: int, 
                          performance_percentile: float = 20.0) -> List[Dict]:
    """
    Select diverse configurations using hybrid approach:
    1. Performance filtering (keep top percentile)
    2. K-Means clustering to identify diverse regions
    3. Select best performer from each cluster
    
    This ensures both diversity (via clustering) and quality (best from each region).
    
    Args:
        records: List of configuration records with metrics
        n_select: Number of configurations to select
        performance_percentile: Keep top X% of configurations by performance
    
    Returns:
        List of selected diverse, high-performing configurations
    """
    if len(records) <= n_select:
        return records
    
    # Step 1: Performance filtering
    sorted_recs = sorted(records, key=lambda x: x.get('validation_loss', float('inf')))
    n_filtered = max(n_select, int(len(records) * performance_percentile / 100))
    n_filtered = min(n_filtered, len(records))
    filtered_recs = sorted_recs[:n_filtered]
    
    print(f"  Performance filtering: {len(records)} -> {len(filtered_recs)} configs "
          f"(top {performance_percentile}%)")
    
    if len(filtered_recs) <= n_select:
        return filtered_recs
    
    # Step 2: Encode configurations to vectors
    # Collect all parameter names
    all_params = set()
    for rec in filtered_recs:
        all_params.update(rec.get('config', {}).keys())
    
    # Remove FLAML_sample_size as it's not a real hyperparameter
    all_params.discard('FLAML_sample_size')
    param_names = sorted(all_params)
    
    # Identify categorical vs numerical parameters
    categorical_params = set()
    for param in param_names:
        # Check if any value is non-numeric
        for rec in filtered_recs:
            value = rec.get('config', {}).get(param)
            if value is not None and not isinstance(value, (int, float)):
                categorical_params.add(param)
                break
    
    # Create label encoders for categorical parameters
    categorical_encoders: Dict[str, LabelEncoder] = {}
    for param in categorical_params:
        values = [rec.get('config', {}).get(param) for rec in filtered_recs 
                 if rec.get('config', {}).get(param) is not None]
        if values:
            encoder = LabelEncoder()
            encoder.fit(values)
            categorical_encoders[param] = encoder
    
    # Encode all configurations
    config_vectors = []
    for rec in filtered_recs:
        vector = encode_config_to_vector(rec.get('config', {}), param_names, 
                                        categorical_encoders)
        config_vectors.append(vector)
    
    X = np.array(config_vectors)
    
    # Handle missing values and normalize
    # Replace sentinel values (-999) with column mean
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        valid_mask = col != -999
        if valid_mask.any():
            col_mean = col[valid_mask].mean()
            col[~valid_mask] = col_mean
    
    # Standardize features (important for distance-based clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Step 3: K-Means clustering + Best performer selection
    print(f"  Running K-Means clustering: {len(filtered_recs)} -> {n_select} configs")
    
    kmeans = KMeans(n_clusters=n_select, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Get cluster labels for each configuration
    cluster_labels = kmeans.labels_
    
    # Select the best performing config from each cluster
    selected_indices = []
    for cluster_id in range(n_select):
        # Get indices of all configs in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = [i for i in range(len(filtered_recs)) if cluster_mask[i]]
        
        if not cluster_indices:
            # Empty cluster (rare but possible) - skip
            continue
        
        # Get the configs in this cluster
        cluster_configs = [(idx, filtered_recs[idx]) for idx in cluster_indices]
        
        # Find the best performer (lowest validation loss)
        best_idx, best_config = min(cluster_configs, 
                                     key=lambda x: x[1].get('validation_loss', float('inf')))
        selected_indices.append(best_idx)
    
    # Select the best configurations from each cluster
    selected_configs = [filtered_recs[idx] for idx in selected_indices]
    
    # Sort by performance
    selected_configs = sorted(selected_configs, 
                             key=lambda x: x.get('validation_loss', float('inf')))
    
    print(f"  Selected {len(selected_configs)} diverse best-performing configurations")
    
    return selected_configs


def save_warm_start_configs(records: List[Dict], n_overall: int, n_per_method: int, 
                            output_file: Path, performance_percentile: float = 20.0) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """
    Save configurations for warm start in Python dictionary format.
    Generates two files:
    1. Absolute top N configurations (by performance)
    2. Representative configurations using hybrid selection:
       - Performance filtering (top X%)
       - K-Means clustering (identify diverse regions)
       - Best performer selection (pick best from each cluster)
    
    Saves two dictionaries: absolute_dict, representative_dict
    """
    # Group by learner
    learner_records: DefaultDict[str, List[Dict]] = defaultdict(list)
    for record in records:
        learner = record.get('learner')
        if learner:
            learner_records[learner].append(record)
    
    # FIX: Properly initialize dictionaries with correct types
    absolute_dict: Dict[str, List[Dict[str, Any]]] = {}
    representative_dict: Dict[str, List[Dict[str, Any]]] = {}
    
    for learner in learner_records:
        # Sort by validation_loss
        sorted_recs = sorted(learner_records[learner], 
                           key=lambda x: x.get('validation_loss', float('inf')))
        
        # ABSOLUTE: Simple top N
        top_n = sorted_recs[:n_per_method]
        
        absolute_list = []
        for idx, record in enumerate(top_n, 1):
            config = record.get('config', {})
            val_loss = record.get('validation_loss')
            absolute_list.append({
                'config': config,
                'rank': idx,
                'metric': val_loss
            })
        absolute_dict[learner] = absolute_list
        
        # REPRESENTATIVE: Hybrid selection (filtering + k-medoids)
        representative_configs = select_diverse_configs(
            learner_records[learner], 
            n_per_method,
            performance_percentile
        )
        
        representative_list = []
        for idx, record in enumerate(representative_configs, 1):
            config = record.get('config', {})
            val_loss = record.get('validation_loss')
            representative_list.append({
                'config': config,
                'rank': idx,
                'metric': val_loss
            })
        representative_dict[learner] = representative_list
    
    # Save ABSOLUTE configs
    absolute_file = output_file.parent / "warm_start_configs_absolute.py"
    _write_warm_start_file(absolute_dict, absolute_file, 
                          "Absolute top N configurations (by performance)")
    
    # Save REPRESENTATIVE configs
    representative_file = output_file.parent / "warm_start_configs_representative.py"
    _write_warm_start_file(representative_dict, representative_file,
                          f"Representative configurations (hybrid selection: top {performance_percentile}% + K-Means + best per cluster)")
    
    # Count total configs
    total_absolute = sum(len(configs) for configs in absolute_dict.values())
    total_representative = sum(len(configs) for configs in representative_dict.values())
    
    print(f"\n{'-'*80}")
    print(f"Absolute configs saved to: {absolute_file}")
    print(f"  - Top {n_per_method} per method")
    print(f"  - {len(absolute_dict)} methods found")
    print(f"  - Total configs: {total_absolute}")
    
    print(f"\nRepresentative configs saved to: {representative_file}")
    print(f"  - Top {n_per_method} diverse per method")
    print(f"  - {len(representative_dict)} methods found")
    print(f"  - Total configs: {total_representative}")
    
    return absolute_dict, representative_dict


def _write_warm_start_file(warm_start_dict: Dict[str, List[Dict[str, Any]]], output_file: Path, description: str):
    """
    Helper function to write warm start configurations to a Python file.
    
    Args:
        warm_start_dict: Dictionary of learner -> list of config dicts
        output_file: Path to output file
        description: Description for the header comment
    """
    with open(output_file, 'w') as f:
        f.write("# Warm start configurations for FLAML\n")
        f.write(f"# {description}\n")
        f.write("# Auto-generated from optimization logs\n\n")
        f.write("warm_start_configs = {\n")
        
        for learner_idx, learner in enumerate(sorted(warm_start_dict.keys())):
            configs = warm_start_dict[learner]
            f.write(f"    # Top {len(configs)} configurations for {learner}\n")
            f.write(f"    '{learner}': [\n")
            
            for cfg in configs:
                # Format config dict as a single line
                config_items = []
                for k, v in cfg['config'].items():
                    if isinstance(v, str):
                        config_items.append(f'"{k}":{repr(v)}')
                    else:
                        config_items.append(f'"{k}":{v}')
                config_str = "{" + ",".join(config_items) + "}"
                
                f.write(f"        {config_str},  # Rank {cfg['rank']}: metric={cfg['metric']:.6f}\n")
            
            f.write("    ],\n")
            # Add blank line between learners except for the last one
            if learner_idx < len(warm_start_dict) - 1:
                f.write("\n")
        
        f.write("}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract best configurations from FLAML optimization logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract top 1 config per learner with default warm start (5 overall + 5 per method)
  python extract_best_configs.py flaml_log.txt
  
  # Extract top 3 configs per learner with custom warm start
  python extract_best_configs.py flaml_log.txt -n 3 --warm-start-overall 10 --warm-start-per-method 5
  
  # Disable warm start configs
  python extract_best_configs.py flaml_log.txt --warm-start-overall 0 --warm-start-per-method 0
  
  # Custom output directory
  python extract_best_configs.py flaml_log.txt -o results/
        """
    )
    
    parser.add_argument('log_file', type=Path,
                        help='Path to FLAML log file (JSON lines format)')
    parser.add_argument('-n', '--n-best', type=int, default=1,
                        help='Number of best configs to extract per learner (default: 1)')
    parser.add_argument('-o', '--output-dir', type=Path, default=None,
                        help='Output directory (default: same as log file)')
    parser.add_argument('--warm-start-overall', type=int, default=5,
                        help='Number of best overall configs for warm start (default: 5)')
    parser.add_argument('--warm-start-per-method', type=int, default=5,
                        help='Number of best configs per method for warm start (default: 5)')
    parser.add_argument('--performance-percentile', type=float, default=20.0,
                        help='Performance filtering: keep top X%% before clustering (default: 20.0)')
    
    args = parser.parse_args()
    
    # Validate input
    if not args.log_file.exists():
        print(f"Error: Log file not found: {args.log_file}")
        return 1
    
    # Set output directory
    if args.output_dir is None:
        output_dir = args.log_file.parent
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filenames
    summary_file = output_dir / "optimization_summary.txt"
    plot_file = output_dir / "optimization_analysis.png"
    search_space_file = output_dir / "search_space_2d.png"
    output_base_path = output_dir / "warm_start_configs.py"  # Base path for generating both files
    
    print(f"Processing: {args.log_file}")
    print(f"Extracting top {args.n_best} config(s) per learner...")
    print()
    
    # Parse log file
    records = parse_flaml_log(args.log_file)
    if not records:
        print("Error: No valid records found in log file")
        return 1
    
    print(f"Parsed {len(records)} records")
    
    # Filter out unknown methods
    records = filter_unknown_methods(records)
    if not records:
        print("Error: No valid records after filtering unknown methods")
        return 1
    
    print(f"Using {len(records)} records with known methods")
    
    # Extract best configs
    best_configs = extract_best_configs(records, args.n_best)
    print(f"Found configurations for {len(best_configs)} learners")
    
    # Analyze optimization progress
    analysis = analyze_optimization_progress(records)
    
    # Generate outputs
    print("\nGenerating outputs...")
    generate_text_summary(analysis, best_configs, summary_file)
    plot_optimization_progress(analysis, plot_file)
    
    # Save warm start configs (now always generated by default)
    absolute_dict: Dict[str, List[Dict[str, Any]]] = {}
    representative_dict: Dict[str, List[Dict[str, Any]]] = {}
    if args.warm_start_overall > 0 or args.warm_start_per_method > 0:
        print("\nGenerating warm start configurations...")
        absolute_dict, representative_dict = save_warm_start_configs(
            records, args.warm_start_overall, args.warm_start_per_method, 
            output_base_path, args.performance_percentile)
        
        # Generate search space visualization
        plot_search_space_2d(records, absolute_dict, representative_dict, search_space_file)
    
    print("\n" + "=" * 80)
    print("QUICK SUMMARY")
    print("=" * 80)
    print(f"Total trials: {analysis['total_trials']}")
    if analysis['validation_losses']:
        print(f"Best validation loss: {min(analysis['validation_losses']):.6f}")
    if analysis['best_overall']:
        print(f"Best learner: {analysis['best_overall'].get('learner')}")
    if analysis['wall_clock_times']:
        print(f"Total time: {max(analysis['wall_clock_times'])/60:.2f} minutes")
    print("\nLearner trial counts:")
    for learner in sorted(analysis['learners'].keys()):
        count = analysis['learners'][learner]['count']
        best = analysis['learners'][learner]['best_loss']
        print(f"  {learner}: {count} trials, best loss = {best:.6f}")
    
    print("\n" + "=" * 80)
    print("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
