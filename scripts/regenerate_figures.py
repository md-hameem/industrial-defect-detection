"""
Regenerate thesis figures with updated model results.
Run this script after updating model training results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
OUTPUTS_DIR = Path("F:/Thesis/outputs")
FIGURES_DIR = OUTPUTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_results():
    """Load all results CSVs."""
    cae = pd.read_csv(OUTPUTS_DIR / "cae_mvtec_results.csv")
    vae = pd.read_csv(OUTPUTS_DIR / "vae_mvtec_results.csv")
    dae = pd.read_csv(OUTPUTS_DIR / "dae_mvtec_results.csv")
    comparison = pd.read_csv(OUTPUTS_DIR / "model_comparison.csv")
    return cae, vae, dae, comparison

def plot_model_comparison_bar(comparison_df):
    """Create grouped bar chart comparing all models."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    categories = comparison_df['category']
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax.bar(x - width, comparison_df['CAE_AUC'], width, label='CAE', color='#3498db', edgecolor='white')
    bars2 = ax.bar(x, comparison_df['VAE_AUC'], width, label='VAE', color='#e74c3c', edgecolor='white')
    bars3 = ax.bar(x + width, comparison_df['DAE_AUC'], width, label='DAE', color='#2ecc71', edgecolor='white')
    
    ax.set_xlabel('Category', fontweight='bold')
    ax.set_ylabel('ROC-AUC Score', fontweight='bold')
    ax.set_title('Model Performance Comparison Across MVTec AD Categories', fontweight='bold', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved model_comparison_bar.png")

def plot_model_heatmap(comparison_df):
    """Create heatmap of model performance."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Prepare data
    data = comparison_df.set_index('category')[['CAE_AUC', 'VAE_AUC', 'DAE_AUC']]
    data.columns = ['CAE', 'VAE', 'DAE']
    
    # Create heatmap
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, center=0.5,
                linewidths=0.5, ax=ax,
                cbar_kws={'label': 'ROC-AUC Score'})
    
    ax.set_title('MVTec AD Performance Heatmap', fontweight='bold', fontsize=16)
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Category', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved model_heatmap.png")

def plot_mean_comparison(comparison_df):
    """Create bar chart of mean AUC per model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    means = {
        'CAE': comparison_df['CAE_AUC'].mean(),
        'VAE': comparison_df['VAE_AUC'].mean(),
        'DAE': comparison_df['DAE_AUC'].mean(),
    }
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(means.keys(), means.values(), color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, means.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Mean ROC-AUC Score', fontweight='bold')
    ax.set_title('Average Performance Across All Categories', fontweight='bold', fontsize=16)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_mean_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved model_mean_comparison.png")

def plot_thesis_fig2(comparison_df):
    """Create thesis figure 2: Combined model comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: Bar chart
    ax1 = axes[0]
    categories = comparison_df['category']
    x = np.arange(len(categories))
    width = 0.25
    
    ax1.bar(x - width, comparison_df['CAE_AUC'], width, label='CAE', color='#3498db')
    ax1.bar(x, comparison_df['VAE_AUC'], width, label='VAE', color='#e74c3c')
    ax1.bar(x + width, comparison_df['DAE_AUC'], width, label='DAE', color='#2ecc71')
    
    ax1.set_xlabel('Category', fontweight='bold')
    ax1.set_ylabel('ROC-AUC Score', fontweight='bold')
    ax1.set_title('(a) Performance by Category', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Right: Mean comparison
    ax2 = axes[1]
    means = {
        'CAE': comparison_df['CAE_AUC'].mean(),
        'VAE': comparison_df['VAE_AUC'].mean(),
        'DAE': comparison_df['DAE_AUC'].mean(),
    }
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax2.bar(means.keys(), means.values(), color=colors, edgecolor='white', linewidth=2)
    
    for bar, val in zip(bars, means.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax2.set_xlabel('Model', fontweight='bold')
    ax2.set_ylabel('Mean ROC-AUC Score', fontweight='bold')
    ax2.set_title('(b) Average Performance', fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.suptitle('Model Performance Comparison on MVTec AD Dataset', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "thesis_fig2_model_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("✓ Saved thesis_fig2_model_comparison.png")

def generate_summary_stats(comparison_df):
    """Print summary statistics."""
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    
    for model in ['CAE', 'VAE', 'DAE']:
        col = f'{model}_AUC'
        print(f"\n{model}:")
        print(f"  Mean AUC: {comparison_df[col].mean():.4f}")
        print(f"  Std AUC:  {comparison_df[col].std():.4f}")
        print(f"  Min AUC:  {comparison_df[col].min():.4f} ({comparison_df.loc[comparison_df[col].idxmin(), 'category']})")
        print(f"  Max AUC:  {comparison_df[col].max():.4f} ({comparison_df.loc[comparison_df[col].idxmax(), 'category']})")
    
    print("\n" + "="*50)

def main():
    print("Loading results...")
    cae, vae, dae, comparison = load_results()
    
    print("\nGenerating figures...")
    plot_model_comparison_bar(comparison)
    plot_model_heatmap(comparison)
    plot_mean_comparison(comparison)
    plot_thesis_fig2(comparison)
    
    generate_summary_stats(comparison)
    
    print("\n✓ All figures regenerated successfully!")
    print(f"  Saved to: {FIGURES_DIR}")

if __name__ == "__main__":
    main()
