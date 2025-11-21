import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_results(results_path='outputs/evaluation_results.json'):
    """Load evaluation results"""
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_speed_comparison(results, save_path='outputs/'):
    """
    Create speed comparison visualization
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Latency comparison
    models = ['Teacher\n(7B)', 'Student\n(350M)']
    latencies = [
        results['teacher']['speed']['avg_latency_seconds'],
        results['student']['speed']['avg_latency_seconds']
    ]
    
    bars = ax[0].bar(models, latencies, color=['#e74c3c', '#3498db'], alpha=0.8, width=0.6)
    ax[0].set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold')
    ax[0].set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    ax[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, latencies):
        height = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width()/2., height,
                  f'{val:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    # Speedup factor
    speedup = results['comparison']['speedup_factor']
    ax[1].bar(['Speedup Factor'], [speedup], color='#2ecc71', alpha=0.8, width=0.4)
    ax[1].set_ylabel('Speedup Factor (x)', fontsize=12, fontweight='bold')
    ax[1].set_title('Student is Faster', fontsize=14, fontweight='bold')
    ax[1].grid(axis='y', alpha=0.3)
    ax[1].text(0, speedup, f'{speedup:.2f}x', ha='center', va='bottom', 
              fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(Path(save_path) / 'speed_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved speed comparison plot")
    plt.close()

def plot_size_memory_comparison(results, save_path='outputs/'):
    """
    Create size and memory comparison visualization
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Model size comparison
    models = ['Teacher', 'Student']
    sizes = [
        results['teacher']['size']['total_parameters'] / 1e9,  # Billions
        results['student']['size']['total_parameters'] / 1e6 / 1000  # Billions
    ]
    
    bars1 = ax[0].bar(models, sizes, color=['#e74c3c', '#3498db'], alpha=0.8, width=0.5)
    ax[0].set_ylabel('Parameters (Billions)', fontsize=12, fontweight='bold')
    ax[0].set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, sizes):
        height = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width()/2., height,
                  f'{val:.2f}B', ha='center', va='bottom', fontweight='bold')
    
    # Add reduction percentage
    reduction = results['comparison']['size_reduction_percent']
    ax[0].text(0.5, max(sizes)*0.7, f'{reduction:.1f}% smaller', 
              ha='center', fontsize=14, fontweight='bold', 
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Memory usage comparison
    memory = [
        results['teacher']['memory']['max_memory_gb'],
        results['student']['memory']['max_memory_gb']
    ]
    
    bars2 = ax[1].bar(models, memory, color=['#e74c3c', '#3498db'], alpha=0.8, width=0.5)
    ax[1].set_ylabel('GPU Memory (GB)', fontsize=12, fontweight='bold')
    ax[1].set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, memory):
        height = bar.get_height()
        ax[1].text(bar.get_x() + bar.get_width()/2., height,
                  f'{val:.2f} GB', ha='center', va='bottom', fontweight='bold')
    
    # Add reduction percentage
    mem_reduction = results['comparison']['memory_reduction_percent']
    ax[1].text(0.5, max(memory)*0.7, f'{mem_reduction:.1f}% less memory', 
              ha='center', fontsize=14, fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(Path(save_path) / 'size_memory_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved size/memory comparison plot")
    plt.close()

def plot_quality_metrics(results, save_path='outputs/'):
    """
    Create quality metrics visualization
    """
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Perplexity comparison (lower is better)
    models = ['Teacher', 'Student']
    perplexities = [
        results['teacher']['perplexity']['perplexity'],
        results['student']['perplexity']['perplexity']
    ]
    
    bars = ax[0].bar(models, perplexities, color=['#2ecc71', '#f39c12'], alpha=0.8, width=0.5)
    ax[0].set_ylabel('Perplexity (lower is better)', fontsize=12, fontweight='bold')
    ax[0].set_title('Model Quality Comparison', fontsize=14, fontweight='bold')
    ax[0].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, perplexities):
        height = bar.get_height()
        ax[0].text(bar.get_x() + bar.get_width()/2., height,
                  f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Quality retention
    retention = (1 / results['comparison']['perplexity_ratio']) * 100
    ax[0].text(0.5, max(perplexities)*0.7, f'{retention:.1f}% quality retained', 
              ha='center', fontsize=14, fontweight='bold',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # MMLU accuracy (if available)
    if results['teacher']['mmlu']['accuracy'] is not None:
        accuracies = [
            results['teacher']['mmlu']['accuracy'] * 100,
            results['student']['mmlu']['accuracy'] * 100
        ]
        
        bars2 = ax[1].bar(models, accuracies, color=['#2ecc71', '#f39c12'], alpha=0.8, width=0.5)
        ax[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax[1].set_title('MMLU Benchmark Accuracy', fontsize=14, fontweight='bold')
        ax[1].set_ylim([0, 100])
        ax[1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars2, accuracies):
            height = bar.get_height()
            ax[1].text(bar.get_x() + bar.get_width()/2., height,
                      f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    else:
        ax[1].text(0.5, 0.5, 'MMLU evaluation\nnot available', 
                  ha='center', va='center', fontsize=14, transform=ax[1].transAxes)
        ax[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(Path(save_path) / 'quality_metrics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved quality metrics plot")
    plt.close()

def plot_pareto_frontier(results, save_path='outputs/'):
    """
    Create speed vs accuracy trade-off plot (Pareto frontier)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get metrics
    teacher_speed = 1.0 / results['teacher']['speed']['avg_latency_seconds']  # samples/sec
    student_speed = 1.0 / results['student']['speed']['avg_latency_seconds']
    
    teacher_quality = 100 - results['teacher']['perplexity']['perplexity']  # Inverse for "quality"
    student_quality = 100 - results['student']['perplexity']['perplexity']
    
    # Plot points
    ax.scatter([teacher_speed], [teacher_quality], s=500, c='#e74c3c', 
              alpha=0.7, label='Teacher (7B)', edgecolors='black', linewidth=2, zorder=3)
    ax.scatter([student_speed], [student_quality], s=500, c='#3498db', 
              alpha=0.7, label='Student (350M)', edgecolors='black', linewidth=2, zorder=3)
    
    # Add labels
    ax.text(teacher_speed, teacher_quality-2, 'Teacher', ha='center', fontsize=12, fontweight='bold')
    ax.text(student_speed, student_quality+2, 'Student', ha='center', fontsize=12, fontweight='bold')
    
    # Draw arrow showing improvement
    ax.annotate('', xy=(student_speed, student_quality), 
               xytext=(teacher_speed, teacher_quality),
               arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.5))
    
    ax.set_xlabel('Inference Speed (samples/sec)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Quality Score (inverse perplexity)', fontsize=13, fontweight='bold')
    ax.set_title('Speed vs Quality Trade-off\n(Pareto Frontier)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(save_path) / 'pareto_frontier.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved Pareto frontier plot")
    plt.close()

def plot_training_curves(save_path='outputs/'):
    """
    Plot training curves from training stats
    """
    try:
        with open('models/training_stats.json', 'r') as f:
            stats = json.load(f)
        
        epochs = [s['epoch'] for s in stats]
        train_loss = [s['train_loss'] for s in stats]
        val_loss = [s['val_loss'] for s in stats]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, train_loss, marker='o', linewidth=2, markersize=8, 
               label='Training Loss', color='#3498db')
        ax.plot(epochs, val_loss, marker='s', linewidth=2, markersize=8, 
               label='Validation Loss', color='#e74c3c')
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(save_path) / 'training_curves.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved training curves plot")
        plt.close()
    except FileNotFoundError:
        print("⚠ Training stats not found, skipping training curves")

def create_summary_dashboard(results, save_path='outputs/'):
    """
    Create a comprehensive summary dashboard
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Knowledge Distillation: Comprehensive Evaluation Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Speed comparison
    ax1 = fig.add_subplot(gs[0, 0])
    speedup = results['comparison']['speedup_factor']
    ax1.bar(['Speedup'], [speedup], color='#2ecc71', alpha=0.8)
    ax1.set_ylabel('Factor', fontweight='bold')
    ax1.set_title('Speed Improvement', fontweight='bold')
    ax1.text(0, speedup/2, f'{speedup:.2f}x\nfaster', ha='center', 
            fontsize=14, fontweight='bold')
    ax1.set_ylim([0, speedup*1.2])
    
    # 2. Size reduction
    ax2 = fig.add_subplot(gs[0, 1])
    size_red = results['comparison']['size_reduction_percent']
    ax2.bar(['Size Reduction'], [size_red], color='#3498db', alpha=0.8)
    ax2.set_ylabel('Percent', fontweight='bold')
    ax2.set_title('Model Size Reduction', fontweight='bold')
    ax2.text(0, size_red/2, f'{size_red:.1f}%\nsmaller', ha='center', 
            fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 100])
    
    # 3. Memory reduction
    ax3 = fig.add_subplot(gs[0, 2])
    mem_red = results['comparison']['memory_reduction_percent']
    ax3.bar(['Memory Reduction'], [mem_red], color='#9b59b6', alpha=0.8)
    ax3.set_ylabel('Percent', fontweight='bold')
    ax3.set_title('Memory Reduction', fontweight='bold')
    ax3.text(0, mem_red/2, f'{mem_red:.1f}%\nless', ha='center', 
            fontsize=14, fontweight='bold')
    ax3.set_ylim([0, 100])
    
    # 4. Latency details
    ax4 = fig.add_subplot(gs[1, :])
    models = ['Teacher (7B)', 'Student (350M)']
    p50 = [results['teacher']['speed']['p50_latency'], 
           results['student']['speed']['p50_latency']]
    p95 = [results['teacher']['speed']['p95_latency'], 
           results['student']['speed']['p95_latency']]
    p99 = [results['teacher']['speed']['p99_latency'], 
           results['student']['speed']['p99_latency']]
    
    x = np.arange(len(models))
    width = 0.25
    
    ax4.bar(x - width, p50, width, label='P50', color='#2ecc71', alpha=0.8)
    ax4.bar(x, p95, width, label='P95', color='#f39c12', alpha=0.8)
    ax4.bar(x + width, p99, width, label='P99', color='#e74c3c', alpha=0.8)
    
    ax4.set_ylabel('Latency (seconds)', fontweight='bold')
    ax4.set_title('Latency Percentiles', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Quality comparison
    ax5 = fig.add_subplot(gs[2, 0])
    retention = (1 / results['comparison']['perplexity_ratio']) * 100
    ax5.bar(['Quality Retained'], [retention], color='#1abc9c', alpha=0.8)
    ax5.set_ylabel('Percent', fontweight='bold')
    ax5.set_title('Quality Retention', fontweight='bold')
    ax5.text(0, retention/2, f'{retention:.1f}%', ha='center', 
            fontsize=14, fontweight='bold')
    ax5.set_ylim([0, 100])
    
    # 6. Key metrics table
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.axis('off')
    
    table_data = [
        ['Metric', 'Teacher', 'Student', 'Improvement'],
        ['Parameters', 
         results['teacher']['size']['parameters_readable'],
         results['student']['size']['parameters_readable'],
         f"{results['comparison']['size_reduction_percent']:.1f}% smaller"],
        ['Latency (avg)',
         f"{results['teacher']['speed']['avg_latency_seconds']:.3f}s",
         f"{results['student']['speed']['avg_latency_seconds']:.3f}s",
         f"{speedup:.2f}x faster"],
        ['Memory',
         f"{results['teacher']['memory']['max_memory_gb']:.2f} GB",
         f"{results['student']['memory']['max_memory_gb']:.2f} GB",
         f"{mem_red:.1f}% less"],
        ['Perplexity',
         f"{results['teacher']['perplexity']['perplexity']:.2f}",
         f"{results['student']['perplexity']['perplexity']:.2f}",
         f"{retention:.1f}% retained"]
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.savefig(Path(save_path) / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved summary dashboard")
    plt.close()

def main():
    """
    Generate all visualizations
    """
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Load results
    results = load_results()
    
    # Create output directory
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Generate all plots
    plot_speed_comparison(results)
    plot_size_memory_comparison(results)
    plot_quality_metrics(results)
    plot_pareto_frontier(results)
    plot_training_curves()
    create_summary_dashboard(results)
    
    print("\n" + "="*60)
    print("✓ ALL VISUALIZATIONS GENERATED!")
    print("="*60)
    print(f"\nPlots saved in: {output_dir}/")

if __name__ == "__main__":
    main()