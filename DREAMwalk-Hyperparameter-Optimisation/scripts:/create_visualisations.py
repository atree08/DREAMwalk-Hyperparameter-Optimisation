"""
Visualization Scripts for DREAMwalk Hyperparameter Optimization
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_optimization_data():
    """Load optimization results"""
    phase1 = pd.read_csv('dreamwalk_optimization/phase1_walk_dynamics.csv')
    phase2 = pd.read_csv('dreamwalk_optimization/phase2_embedding_capacity.csv')
    return phase1, phase2

def plot_convergence_history(phase1, phase2):
    """Plot optimization convergence over trials"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Phase 1
    trials1 = phase1['number']
    values1 = phase1['value']
    best1 = phase1['value'].cummax()
    
    ax1.plot(trials1, values1, 'o-', alpha=0.6, label='Trial Score')
    ax1.plot(trials1, best1, 'r-', linewidth=2, label='Best Score')
    ax1.axhline(y=0.9578, color='gray', linestyle='--', label='Baseline (Run 3)')
    ax1.set_xlabel('Trial Number', fontsize=12)
    ax1.set_ylabel('Combined Score (AUROC + AUPR)/2', fontsize=12)
    ax1.set_title('Phase 1: Walk Dynamics Optimization', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Phase 2
    trials2 = phase2['number']
    values2 = phase2['value']
    best2 = phase2['value'].cummax()
    
    ax2.plot(trials2, values2, 'o-', alpha=0.6, label='Trial Score')
    ax2.plot(trials2, best2, 'r-', linewidth=2, label='Best Score')
    ax2.set_xlabel('Trial Number', fontsize=12)
    ax2.set_ylabel('Combined Score (AUROC + AUPR)/2', fontsize=12)
    ax2.set_title('Phase 2: Embedding Capacity Optimization', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_convergence.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: optimization_convergence.png")
    plt.close()

def plot_parameter_sensitivity(phase1):
    """Plot how each parameter affects performance"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    params = ['params_num_walks', 'params_walk_length', 'params_tp_factor', 
              'params_p', 'params_q']
    titles = ['Number of Walks', 'Walk Length', 'Teleportation Factor', 
              'Return Parameter (p)', 'In-Out Parameter (q)']
    
    for idx, (param, title) in enumerate(zip(params, titles)):
        if param not in phase1.columns:
            continue
            
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        
        # Group by parameter value
        grouped = phase1.groupby(param)['value'].agg(['mean', 'std', 'count'])
        
        if param in ['params_tp_factor']:
            # Continuous parameter - use scatter + trend
            x = phase1[param]
            y = phase1['value']
            ax.scatter(x, y, alpha=0.5, s=100)
            
            # Add trend line
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_smooth, p(x_smooth), 'r--', alpha=0.8, linewidth=2)
            
            ax.set_xlabel(title, fontsize=11)
            ax.set_ylabel('Combined Score', fontsize=11)
        else:
            # Categorical parameter - use bar plot
            x_pos = range(len(grouped))
            ax.bar(x_pos, grouped['mean'], yerr=grouped['std'], 
                   capsize=5, alpha=0.7, color='steelblue')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(grouped.index, rotation=45)
            ax.set_xlabel(title, fontsize=11)
            ax.set_ylabel('Mean Combined Score', fontsize=11)
            
            # Add sample counts
            for i, (idx_val, row) in enumerate(grouped.iterrows()):
                ax.text(i, row['mean'] + row['std'] + 0.001, 
                       f'n={int(row["count"])}', 
                       ha='center', fontsize=9, color='gray')
        
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: parameter_sensitivity.png")
    plt.close()

def plot_embedding_capacity_comparison(phase2):
    """Compare dimension and window size effects"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Dimension comparison
    dim_grouped = phase2.groupby('params_dimension')['value'].agg(['mean', 'std', 'count'])
    x1 = range(len(dim_grouped))
    ax1.bar(x1, dim_grouped['mean'], yerr=dim_grouped['std'], 
            capsize=5, alpha=0.7, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_xticks(x1)
    ax1.set_xticklabels(dim_grouped.index)
    ax1.set_xlabel('Embedding Dimension', fontsize=12)
    ax1.set_ylabel('Mean Combined Score', fontsize=12)
    ax1.set_title('Effect of Embedding Dimension', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Window size comparison
    win_grouped = phase2.groupby('params_window_size')['value'].agg(['mean', 'std', 'count'])
    x2 = range(len(win_grouped))
    ax2.bar(x2, win_grouped['mean'], yerr=win_grouped['std'], 
            capsize=5, alpha=0.7, color=['#95E1D3', '#F38181', '#AA96DA'])
    ax2.set_xticks(x2)
    ax2.set_xticklabels(win_grouped.index)
    ax2.set_xlabel('Context Window Size', fontsize=12)
    ax2.set_ylabel('Mean Combined Score', fontsize=12)
    ax2.set_title('Effect of Context Window', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('embedding_capacity_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: embedding_capacity_comparison.png")
    plt.close()

def plot_teleportation_impact(phase1):
    """Detailed analysis of teleportation factor"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot with color gradient
    tp = phase1['params_tp_factor']
    score = phase1['value']
    
    scatter = ax1.scatter(tp, score, c=score, cmap='RdYlGn', s=100, alpha=0.7)
    ax1.set_xlabel('Teleportation Factor', fontsize=12)
    ax1.set_ylabel('Combined Score', fontsize=12)
    ax1.set_title('Teleportation Factor vs Performance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Score')
    
    # Add trend line
    z = np.polyfit(tp, score, 2)
    p = np.poly1d(z)
    tp_smooth = np.linspace(tp.min(), tp.max(), 100)
    ax1.plot(tp_smooth, p(tp_smooth), 'r--', linewidth=2, alpha=0.8, label='Trend')
    ax1.legend()
    
    # Histogram of best performers
    top_20_pct = phase1.nlargest(int(len(phase1) * 0.2), 'value')
    ax2.hist(top_20_pct['params_tp_factor'], bins=15, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(top_20_pct['params_tp_factor'].mean(), color='red', 
                linestyle='--', linewidth=2, label=f'Mean: {top_20_pct["params_tp_factor"].mean():.3f}')
    ax2.set_xlabel('Teleportation Factor', fontsize=12)
    ax2.set_ylabel('Frequency (Top 20% Trials)', fontsize=12)
    ax2.set_title('Optimal Teleportation Range', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('teleportation_impact.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: teleportation_impact.png")
    plt.close()

def plot_baseline_vs_optimized():
    """Compare baseline vs optimized results"""
    import json
    
    # Load optimized results
    with open('final_optimized_results.json', 'r') as f:
        opt_data = json.load(f)
    
    # Baseline (Run 3)
    baseline = {
        'AUROC': 0.9578,
        'AUPR': 0.9083,
        'Accuracy': 0.9093
    }
    
    optimized = {
        'AUROC': opt_data['mean']['auroc'],
        'AUPR': opt_data['mean']['aupr'],
        'Accuracy': opt_data['mean']['accuracy']
    }
    
    metrics = list(baseline.keys())
    baseline_vals = [baseline[m] for m in metrics]
    optimized_vals = [optimized[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (Run 3)', 
                   alpha=0.8, color='#FF6B6B')
    bars2 = ax.bar(x + width/2, optimized_vals, width, label='Optimized', 
                   alpha=0.8, color='#4ECDC4')
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Baseline vs Optimized Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
    
    # Add improvement percentages
    for i, metric in enumerate(metrics):
        improvement = ((optimized_vals[i] - baseline_vals[i]) / baseline_vals[i]) * 100
        ax.text(i, max(baseline_vals[i], optimized_vals[i]) + 0.01,
               f'+{improvement:.1f}%',
               ha='center', fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('baseline_vs_optimized.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: baseline_vs_optimized.png")
    plt.close()

def create_all_visualizations():
    """Generate all visualization figures"""
    print("="*70)
    print("CREATING VISUALIZATION FIGURES")
    print("="*70)
    
    # Load data
    print("\nLoading optimization data...")
    phase1, phase2 = load_optimization_data()
    
    # Generate figures
    print("\n1. Plotting convergence history...")
    plot_convergence_history(phase1, phase2)
    
    print("\n2. Plotting parameter sensitivity...")
    plot_parameter_sensitivity(phase1)
    
    print("\n3. Plotting embedding capacity comparison...")
    plot_embedding_capacity_comparison(phase2)
    
    print("\n4. Plotting teleportation impact...")
    plot_teleportation_impact(phase1)
    
    print("\n5. Plotting baseline vs optimized comparison...")
    try:
        plot_baseline_vs_optimized()
    except FileNotFoundError:
        print("   ⚠️  Skipped: Run final 10-fold CV first")
    
    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  • optimization_convergence.png")
    print("  • parameter_sensitivity.png")
    print("  • embedding_capacity_comparison.png")
    print("  • teleportation_impact.png")
    print("  • baseline_vs_optimized.png")

if __name__ == '__main__':
    create_all_visualizations()