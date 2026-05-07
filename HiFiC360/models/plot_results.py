import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

def load_and_transpose_csv(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    
    df_t = df.T
    
    df_t.reset_index(inplace=True)
    df_t.rename(columns={'index': 'model'}, inplace=True)
    
    numeric_cols = ['psnr', 'ws-psnr', 'ssim', 'ws_ssim', 'mse', 'ws-mse', 'bpp_real']
    for col in numeric_cols:
        if col in df_t.columns:
            df_t[col] = pd.to_numeric(df_t[col], errors='coerce')
            
    return df_t

def plot_results(csv_path, output_file=None):
    df = load_and_transpose_csv(csv_path)
    
    metrics = ['ws-psnr', 'ws_ssim']
    titles = ['WS-PSNR (dB)', 'WS-SSIM']
    
    groups = df['group'].unique()
    colors = plt.cm.tab10(range(len(groups)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    group_style = {}
    for i, group in enumerate(groups):
        group_style[group] = {
            'color': colors[i % len(colors)],
            'marker': markers[i % len(markers)]
        }

    for col, metric in enumerate(metrics):
        fig, ax = plt.subplots(figsize=(6, 4))
        
        if metric not in df.columns:
            ax.text(0.5, 0.5, f'{metric} not found', ha='center', va='center')
            continue
            
        for group in groups:
            group_data = df[df['group'] == group].sort_values('bpp_real')
            
            style = group_style[group]
            ax.plot(group_data['bpp_real'], group_data[metric], 
                   label=group,
                   color=style['color'],
                   marker=style['marker'],
                   markersize=8,
                   linewidth=2)
        
        title = titles[col]
        ax.set_xlabel('bitrate (bpp)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        
        ax.set_xlim(0.125, 0.425)
        
        hardcoded_group = 'HiFiC360 (Ours)'  
        if hardcoded_group in df['group'].values:
            group_data_hc = df[df['group'] == hardcoded_group]
            if not group_data_hc[metric].isna().all():
                min_y = group_data_hc[metric].min()
                max_y = group_data_hc[metric].max()
                margin = (max_y - min_y) * 0.05 if max_y > min_y else 0.1
                ax.set_ylim(min_y - margin, max_y + margin)

        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend(fontsize=11, frameon=True)
        
        fig.tight_layout()
        
        if output_file:
            base, ext = os.path.splitext(output_file)
            metric_output_file = f"{base}_{metric}{ext}"
            plt.savefig(metric_output_file, bbox_inches='tight')
            print(f"Plot saved to {metric_output_file}")
            
    if not output_file:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot evaluation results from CSV.')
    parser.add_argument('csv_file', help='Path to the results CSV file.')
    parser.add_argument('--output', '-o', default='results_plot.pdf', help='Output image file.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found.")
        sys.exit(1)
        
    plot_results(args.csv_file, args.output)
