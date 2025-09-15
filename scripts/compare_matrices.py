#!/usr/bin/env python3
"""
Compare distance and time matrices before and after OSRM integration.

Usage:
  python3 scripts/compare_matrices.py \
      --old-dir data/private/active \
      --new-dir data/private/active_osrm \
      --location-index data/private/active/location_index.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any

def load_matrix(file_path: Path) -> np.ndarray:
    """Load matrix from .npz file."""
    try:
        data = np.load(file_path)
        return data['matrix']
    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path}: {e}")

def matrix_stats(matrix: np.ndarray, name: str) -> Dict[str, Any]:
    """Calculate comprehensive matrix statistics."""
    # Remove diagonal (self-distances)
    non_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    non_zero = non_diag[non_diag > 0]
    
    total_pairs = matrix.size - matrix.shape[0]  # Exclude diagonal
    valid_pairs = np.sum(matrix > 0) - matrix.shape[0]  # Exclude diagonal zeros
    coverage = (valid_pairs / total_pairs * 100) if total_pairs > 0 else 0
    
    stats = {
        'name': name,
        'shape': matrix.shape,
        'total_pairs': total_pairs,
        'valid_pairs': valid_pairs,
        'coverage_percent': coverage,
        'min_nonzero': float(np.min(non_zero)) if len(non_zero) > 0 else 0,
        'max': float(np.max(non_zero)) if len(non_zero) > 0 else 0,
        'mean': float(np.mean(non_zero)) if len(non_zero) > 0 else 0,
        'median': float(np.median(non_zero)) if len(non_zero) > 0 else 0,
        'std': float(np.std(non_zero)) if len(non_zero) > 0 else 0,
        'zeros': int(np.sum(matrix == 0) - matrix.shape[0]),  # Exclude diagonal
    }
    return stats

def compare_matrices(old_matrix: np.ndarray, new_matrix: np.ndarray, 
                    locations_df: pd.DataFrame, matrix_type: str) -> Dict[str, Any]:
    """Compare two matrices and generate detailed analysis."""
    
    print(f"\n=== {matrix_type.upper()} MATRIX COMPARISON ===")
    
    # Basic stats
    old_stats = matrix_stats(old_matrix, f"Old {matrix_type}")
    new_stats = matrix_stats(new_matrix, f"New {matrix_type}")
    
    print(f"\n📊 Coverage Comparison:")
    print(f"  Old: {old_stats['coverage_percent']:.1f}% ({old_stats['valid_pairs']}/{old_stats['total_pairs']} pairs)")
    print(f"  New: {new_stats['coverage_percent']:.1f}% ({new_stats['valid_pairs']}/{new_stats['total_pairs']} pairs)")
    print(f"  Improvement: +{new_stats['coverage_percent'] - old_stats['coverage_percent']:.1f} percentage points")
    
    unit = "miles" if matrix_type == "distance" else "minutes"
    print(f"\n📈 Value Ranges ({unit}):")
    print(f"  Old: {old_stats['min_nonzero']:.1f} - {old_stats['max']:.1f} (mean: {old_stats['mean']:.1f})")
    print(f"  New: {new_stats['min_nonzero']:.1f} - {new_stats['max']:.1f} (mean: {new_stats['mean']:.1f})")
    
    # Identify changes
    old_nonzero = (old_matrix > 0) & ~np.eye(old_matrix.shape[0], dtype=bool)
    new_nonzero = (new_matrix > 0) & ~np.eye(new_matrix.shape[0], dtype=bool)
    
    # New routes (0 -> positive)
    new_routes = (~old_nonzero) & new_nonzero
    new_routes_count = np.sum(new_routes)
    
    # Lost routes (positive -> 0) 
    lost_routes = old_nonzero & (~new_nonzero)
    lost_routes_count = np.sum(lost_routes)
    
    # Changed routes (both positive, different values)
    both_nonzero = old_nonzero & new_nonzero
    significant_changes = both_nonzero & (np.abs(old_matrix - new_matrix) > (0.1 * old_matrix))
    changed_routes_count = np.sum(significant_changes)
    
    print(f"\n🔄 Route Changes:")
    print(f"  New routes: {new_routes_count}")
    print(f"  Lost routes: {lost_routes_count}")
    print(f"  Significantly changed routes: {changed_routes_count}")
    
    # Sample new routes
    if new_routes_count > 0:
        print(f"\n🆕 Sample New Routes:")
        new_indices = np.where(new_routes)
        sample_size = min(5, len(new_indices[0]))
        for idx in range(sample_size):
            i, j = new_indices[0][idx], new_indices[1][idx]
            from_name = locations_df.iloc[i]['name'] if i < len(locations_df) else f"ID_{i}"
            to_name = locations_df.iloc[j]['name'] if j < len(locations_df) else f"ID_{j}"
            value = new_matrix[i, j]
            print(f"    {from_name} -> {to_name}: {value:.1f} {unit}")
    
    # Sample significant changes
    if changed_routes_count > 0:
        print(f"\n📈 Sample Route Changes:")
        changed_indices = np.where(significant_changes)
        sample_size = min(5, len(changed_indices[0]))
        for idx in range(sample_size):
            i, j = changed_indices[0][idx], changed_indices[1][idx]
            from_name = locations_df.iloc[i]['name'] if i < len(locations_df) else f"ID_{i}"
            to_name = locations_df.iloc[j]['name'] if j < len(locations_df) else f"ID_{j}"
            old_val = old_matrix[i, j]
            new_val = new_matrix[i, j]
            change_pct = ((new_val - old_val) / old_val * 100) if old_val > 0 else 0
            print(f"    {from_name} -> {to_name}: {old_val:.1f} -> {new_val:.1f} {unit} ({change_pct:+.1f}%)")
    
    return {
        'old_stats': old_stats,
        'new_stats': new_stats,
        'new_routes': new_routes_count,
        'lost_routes': lost_routes_count,
        'changed_routes': changed_routes_count,
        'coverage_improvement': new_stats['coverage_percent'] - old_stats['coverage_percent']
    }

def create_comparison_plots(old_dist: np.ndarray, new_dist: np.ndarray,
                          old_time: np.ndarray, new_time: np.ndarray,
                          output_dir: Path):
    """Create comparison plots."""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Matrix Comparison: Old vs New (OSRM)', fontsize=16)
    
    # Distance matrices heatmaps
    def plot_matrix_heatmap(matrix, title, ax, vmax=None):
        # Sample for visualization if matrix is large
        if matrix.shape[0] > 50:
            idx = np.random.choice(matrix.shape[0], 50, replace=False)
            matrix_sample = matrix[np.ix_(idx, idx)]
        else:
            matrix_sample = matrix
            
        mask = matrix_sample == 0
        sns.heatmap(matrix_sample, mask=mask, cmap='viridis', ax=ax, 
                   cbar_kws={'label': 'Miles' if 'Distance' in title else 'Minutes'},
                   vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel('Destination')
        ax.set_ylabel('Origin')
    
    # Set consistent scales
    dist_vmax = max(np.max(old_dist[old_dist > 0]), np.max(new_dist[new_dist > 0]))
    time_vmax = max(np.max(old_time[old_time > 0]), np.max(new_time[new_time > 0]))
    
    plot_matrix_heatmap(old_dist, 'Old Distance Matrix', axes[0,0], dist_vmax)
    plot_matrix_heatmap(new_dist, 'New Distance Matrix', axes[0,1], dist_vmax)
    
    # Distance difference
    diff_dist = new_dist - old_dist
    im = axes[0,2].imshow(diff_dist, cmap='RdBu_r', vmin=-50, vmax=50)
    axes[0,2].set_title('Distance Difference (New - Old)')
    plt.colorbar(im, ax=axes[0,2], label='Miles')
    
    # Time matrices
    plot_matrix_heatmap(old_time, 'Old Time Matrix', axes[1,0], time_vmax)
    plot_matrix_heatmap(new_time, 'New Time Matrix', axes[1,1], time_vmax)
    
    # Time difference
    diff_time = new_time - old_time
    im2 = axes[1,2].imshow(diff_time, cmap='RdBu_r', vmin=-30, vmax=30)
    axes[1,2].set_title('Time Difference (New - Old)')
    plt.colorbar(im2, ax=axes[1,2], label='Minutes')
    
    plt.tight_layout()
    
    plot_file = output_dir / 'matrix_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"📊 Saved comparison plot: {plot_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare distance/time matrices")
    parser.add_argument("--old-dir", required=True, help="Directory with old matrices")
    parser.add_argument("--new-dir", required=True, help="Directory with new matrices") 
    parser.add_argument("--location-index", required=True, help="Path to location_index.csv")
    parser.add_argument("--output-dir", default="./matrix_comparison", help="Output directory for plots")
    
    args = parser.parse_args()
    
    old_dir = Path(args.old_dir)
    new_dir = Path(args.new_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load location index
    locations_df = pd.read_csv(args.location_index)
    print(f"📍 Loaded {len(locations_df)} locations")
    
    # Load matrices
    print("📁 Loading matrices...")
    old_dist = load_matrix(old_dir / "distance_miles_matrix.npz")
    new_dist = load_matrix(new_dir / "distance_miles_matrix.npz")
    old_time = load_matrix(old_dir / "time_minutes_matrix.npz")
    new_time = load_matrix(new_dir / "time_minutes_matrix.npz")
    
    print(f"✅ Loaded matrices. Shape: {old_dist.shape}")
    
    # Compare distance matrices
    dist_comparison = compare_matrices(old_dist, new_dist, locations_df, "distance")
    
    # Compare time matrices  
    time_comparison = compare_matrices(old_time, new_time, locations_df, "time")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"📋 SUMMARY")
    print(f"="*60)
    print(f"Distance Matrix:")
    print(f"  Coverage improvement: +{dist_comparison['coverage_improvement']:.1f} percentage points")
    print(f"  New routes added: {dist_comparison['new_routes']}")
    print(f"  Routes modified: {dist_comparison['changed_routes']}")
    
    print(f"\nTime Matrix:")
    print(f"  Coverage improvement: +{time_comparison['coverage_improvement']:.1f} percentage points")
    print(f"  New routes added: {time_comparison['new_routes']}")
    print(f"  Routes modified: {time_comparison['changed_routes']}")
    
    # Create plots
    try:
        create_comparison_plots(old_dist, new_dist, old_time, new_time, output_dir)
    except Exception as e:
        print(f"⚠️  Plot generation failed: {e}")
    
    # Save detailed comparison report
    report = {
        'distance_comparison': dist_comparison,
        'time_comparison': time_comparison,
        'summary': {
            'total_locations': len(locations_df),
            'matrix_shape': old_dist.shape,
            'distance_coverage_improvement': dist_comparison['coverage_improvement'],
            'time_coverage_improvement': time_comparison['coverage_improvement'],
        }
    }
    
    import json
    report_file = output_dir / 'comparison_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved: {report_file}")
    print(f"✅ Comparison complete!")

if __name__ == "__main__":
    main()