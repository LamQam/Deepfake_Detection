import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Your actual data from the paper
data = {
    'DeepfakeTIMIT': {
        'XceptionNet': {'accuracy': 79.17, 'f1': 0.7692, 'auc': 0.9488},
        'EfficientNetB4': {'accuracy': 93.75, 'f1': 0.9126, 'auc': 0.9453},
        'EfficientNetB7': {'accuracy': 95.83, 'f1': 0.9796, 'auc': 1.0000},
        'MesoNet': {'accuracy': 100.0, 'f1': 1.0000, 'auc': 1.0000},
        'ResNet152': {'accuracy': 93.75, 'f1': 0.9600, 'auc': 0.9852}
    },
    'WildDeepfake': {
        'XceptionNet': {'accuracy': 40.46, 'f1': 0.3705, 'auc': 0.3183},
        'EfficientNetB4': {'accuracy': 67.17, 'f1': 0.6667, 'auc': 0.7673},
        'ResNet152': {'accuracy': 69.21, 'f1': 0.6920, 'auc': 0.7721},
        'MesoNet': {'accuracy': 68.54, 'f1': 0.6801, 'auc': 0.7105}
    }
}

# Model parameters (approximate)
model_params = {
    'XceptionNet': 22.9,  # Million parameters
    'EfficientNetB4': 19.3,
    'EfficientNetB7': 66.3,
    'MesoNet': 0.08,  # 80K parameters
    'ResNet152': 60.2
}

plt.style.use('seaborn-v0_8')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E5B3C']

# 1. SIDE-BY-SIDE PERFORMANCE COMPARISON


def create_performance_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    models = list(data['DeepfakeTIMIT'].keys())
    metrics = ['accuracy', 'f1', 'auc']
    metric_names = ['Accuracy (%)', 'F1-Score', 'AUC']

    x = np.arange(len(models))
    width = 0.35

    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        deepfake_vals = [data['DeepfakeTIMIT'][model][metric]
                         for model in models]
        wild_vals = [data['WildDeepfake'][model][metric]
                     if model in data['WildDeepfake'] else 0 for model in models]

        # Convert accuracy to percentage for display
        if metric == 'accuracy':
            deepfake_vals = deepfake_vals  # Already in percentage
            wild_vals = wild_vals

        bars1 = axes[i].bar(x - width/2, deepfake_vals, width, label='DeepfakeTIMIT',
                            color='#2E86AB', alpha=0.8)
        bars2 = axes[i].bar(x + width/2, wild_vals, width, label='WildDeepfake',
                            color='#A23B72', alpha=0.8)

        axes[i].set_xlabel('Models')
        axes[i].set_ylabel(metric_name)
        axes[i].set_title(f'Model Comparison: {metric_name}')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(models, rotation=45, ha='right')
        axes[i].legend()
        axes[i].grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{height:.1f}', ha='center', va='bottom', fontsize=9)

        for bar in bars2:
            height = bar.get_height()
            if height > 0:  # Only add label if there's data
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

# 2. RADAR CHART FOR MULTI-METRIC COMPARISON


def create_radar_chart():
    from math import pi

    # Select models for radar chart (exclude EfficientNetB7 due to missing WildDeepfake data)
    selected_models = ['XceptionNet', 'EfficientNetB4', 'ResNet152', 'MesoNet']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(
        16, 8), subplot_kw=dict(projection='polar'))

    # Metrics for radar chart
    metrics = ['accuracy', 'f1', 'auc']
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]  # Complete the circle

    for dataset, ax in zip(['DeepfakeTIMIT', 'WildDeepfake'], [ax1, ax2]):
        for i, model in enumerate(selected_models):
            if model in data[dataset]:
                values = [data[dataset][model][metric] for metric in metrics]
                # Normalize accuracy to 0-1 scale for radar chart
                # Convert accuracy percentage to decimal
                values[0] = values[0] / 100
                values += values[:1]  # Complete the circle

                ax.plot(angles, values, 'o-', linewidth=2,
                        label=model, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['Accuracy', 'F1-Score', 'AUC'])
        ax.set_ylim(0, 1)
        ax.set_title(f'{dataset}', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)

    plt.tight_layout()
    plt.show()

# 3. EFFICIENCY vs PERFORMANCE SCATTER PLOT


def create_efficiency_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    datasets = ['DeepfakeTIMIT', 'WildDeepfake']

    for i, dataset in enumerate(datasets):
        ax = ax1 if i == 0 else ax2

        models_in_dataset = [
            m for m in model_params.keys() if m in data[dataset]]
        params = [model_params[model] for model in models_in_dataset]
        accuracies = [data[dataset][model]['accuracy']
                      for model in models_in_dataset]

        scatter = ax.scatter(params, accuracies, s=100,
                             alpha=0.7, c=colors[:len(models_in_dataset)])

        # Add model labels
        for j, model in enumerate(models_in_dataset):
            ax.annotate(model, (params[j], accuracies[j]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax.set_xlabel('Parameters (Millions)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{dataset}: Efficiency vs Performance')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')  # Log scale for parameters

    plt.tight_layout()
    plt.show()

# 4. PERFORMANCE DROP ANALYSIS


def create_performance_drop():
    models_both_datasets = [
        m for m in data['DeepfakeTIMIT'].keys() if m in data['WildDeepfake']]

    drops = []
    for model in models_both_datasets:
        deepfake_acc = data['DeepfakeTIMIT'][model]['accuracy']
        wild_acc = data['WildDeepfake'][model]['accuracy']
        drop = deepfake_acc - wild_acc
        drops.append(drop)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models_both_datasets, drops, color=[
                  '#C73E1D' if d > 30 else '#F18F01' if d > 20 else '#2E86AB' for d in drops])

    ax.set_ylabel('Performance Drop (%)')
    ax.set_title('Accuracy Drop: DeepfakeTIMIT to WildDeepfake')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    # Add value labels
    for bar, drop in zip(bars, drops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{drop:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

# 5. HEATMAP OF ALL METRICS


def create_performance_heatmap():
    # Prepare data for heatmap
    heatmap_data = []
    models = ['XceptionNet', 'EfficientNetB4',
              'ResNet152', 'MesoNet']  # Models in both datasets

    for model in models:
        row = []
        # DeepfakeTIMIT metrics
        row.extend([
            data['DeepfakeTIMIT'][model]['accuracy'],
            data['DeepfakeTIMIT'][model]['f1'] * 100,  # Convert to percentage
            data['DeepfakeTIMIT'][model]['auc'] * 100
        ])
        # WildDeepfake metrics
        row.extend([
            data['WildDeepfake'][model]['accuracy'],
            data['WildDeepfake'][model]['f1'] * 100,
            data['WildDeepfake'][model]['auc'] * 100
        ])
        heatmap_data.append(row)

    columns = ['DT_Accuracy', 'DT_F1', 'DT_AUC',
               'WD_Accuracy', 'WD_F1', 'WD_AUC']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data,
                xticklabels=columns,
                yticklabels=models,
                annot=True,
                fmt='.1f',
                cmap='RdYlBu_r',
                ax=ax)

    ax.set_title('Performance Heatmap: All Models and Metrics')
    plt.tight_layout()
    plt.show()


# Execute all visualizations
if __name__ == "__main__":
    print("Generating Model Performance Visualizations...")

    print("\n1. Side-by-side Performance Comparison")
    create_performance_comparison()

    print("\n2. Radar Chart Multi-metric Comparison")
    create_radar_chart()

    print("\n3. Efficiency vs Performance Analysis")
    create_efficiency_plot()

    print("\n4. Performance Drop Analysis")
    create_performance_drop()

    print("\n5. Performance Heatmap")
    create_performance_heatmap()

    print("\nVisualization Summary:")
    print("- Bar charts: Direct metric comparison")
    print("- Radar charts: Multi-dimensional performance view")
    print("- Scatter plots: Efficiency analysis")
    print("- Drop analysis: Dataset difficulty assessment")
    print("- Heatmaps: Comprehensive overview")
