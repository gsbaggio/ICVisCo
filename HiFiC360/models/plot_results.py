import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os

def load_and_transpose_csv(csv_path):
    # Read the CSV where rows are metrics and columns are models
    df = pd.read_csv(csv_path, index_col=0)
    
    # Transpose so models are rows and metrics are columns
    df_t = df.T
    
    # Reset index to get model names as a column
    df_t.reset_index(inplace=True)
    df_t.rename(columns={'index': 'model'}, inplace=True)
    
    # Convert numeric columns
    numeric_cols = ['psnr', 'ws-psnr', 'ssim', 'ws_ssim', 'mse', 'ws-mse', 'bpp_real']
    for col in numeric_cols:
        if col in df_t.columns:
            df_t[col] = pd.to_numeric(df_t[col], errors='coerce')
            
    return df_t

def plot_results(csv_path, output_file=None):
    df = load_and_transpose_csv(csv_path)
    
    # Define metrics layout
    metrics_grid = [
        ['psnr', 'ssim', 'mse'],      # Top row: Normal metrics
        ['ws-psnr', 'ws_ssim', 'ws-mse'] # Bottom row: WS metrics
    ]
    
    titles = [
        ['PSNR', 'SSIM', 'MSE'],
        ['WS-PSNR', 'WS-SSIM', 'WS-MSE']
    ]
    
    # Setup plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Get unique groups and assign colors/markers
    groups = df['group'].unique()
    colors = plt.cm.tab10(range(len(groups)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    group_style = {}
    for i, group in enumerate(groups):
        group_style[group] = {
            'color': colors[i % len(colors)],
            'marker': markers[i % len(markers)]
        }

    # Plot each metric
    for row in range(2):
        for col in range(3):
            metric = metrics_grid[row][col]
            ax = axes[row, col]
            
            if metric not in df.columns:
                ax.text(0.5, 0.5, f'{metric} not found', ha='center', va='center')
                continue
                
            # Plot each group
            for group in groups:
                group_data = df[df['group'] == group].sort_values('bpp_real')
                
                style = group_style[group]
                ax.plot(group_data['bpp_real'], group_data[metric], 
                       label=group,
                       color=style['color'],
                       marker=style['marker'],
                       markersize=8,
                       linewidth=2)
            
            # Formatting
            ax.set_title(titles[row][col] + (' ↓' if 'mse' in metric.lower() else ' ↑'), fontsize=14)
            ax.set_xlabel('bpp', fontsize=12)
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            
            # Invert y-axis for MSE
            # if 'mse' in metric.lower():
            #     ax.invert_yaxis()

    # Create a single legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
              ncol=len(groups), fontsize=14, frameon=True)

    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot evaluation results from CSV.')
    parser.add_argument('csv_file', help='Path to the results CSV file.')
    parser.add_argument('--output', '-o', default='results_plot.png', help='Output image file.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found.")
        sys.exit(1)
        
    plot_results(args.csv_file, args.output)
