import pandas as pd
import bjontegaard as bd
import argparse
import sys
import os

def load_and_transpose_csv(csv_path):
    # Read the CSV where rows are metrics and columns are models
    df = pd.read_csv(csv_path, index_col=0)
    # Transpose so models are rows and metrics are columns
    df_t = df.T
    df_t.reset_index(inplace=True)
    df_t.rename(columns={'index': 'model'}, inplace=True)
    
    # Convert numeric columns
    numeric_cols = ['psnr', 'ws-psnr', 'ssim', 'ws_ssim', 'mse', 'ws-mse', 'bpp_real']
    for col in numeric_cols:
        if col in df_t.columns:
            df_t[col] = pd.to_numeric(df_t[col], errors='coerce')
            
    return df_t

def analyze_bjontegaard(csv_path):
    df = load_and_transpose_csv(csv_path)
    
    baseline_group = 'LPIPS'
    if baseline_group not in df['group'].values:
        print(f"Baseline group '{baseline_group}' not found in the dataset.")
        return
        
    metric = 'ws-psnr'
    
    baseline_data = df[df['group'] == baseline_group].sort_values('bpp_real')
    baseline_rate = baseline_data['bpp_real'].values
    if metric not in baseline_data.columns:
        print(f"Metric '{metric}' not found in the baseline data.")
        return
    baseline_m = baseline_data[metric].values
    
    groups = [g for g in df['group'].unique() if g != baseline_group]
    
    print(f"{'Group':<20} | {'BD-Rate (%)':<15} | {'BD-WS-PSNR (dB)':<15}")
    print("-" * 55)
    
    for group in groups:
        group_data = df[df['group'] == group].sort_values('bpp_real')
        group_rate = group_data['bpp_real'].values
        
        print(f"{group:<20} | ", end="")
        
        if len(group_data) == 0 or metric not in group_data.columns:
            print(f"{'N/A':<15} | {'N/A':<15}")
            continue
            
        group_m = group_data[metric].values
        
        try:
            # Calculate BD-Rate (delta bit rate)
            bd_rate_val = bd.bd_rate(baseline_rate, baseline_m, group_rate, group_m, method='akima', min_overlap=0)
            
            # Calculate BD-PSNR (delta distortion)
            bd_psnr_val = bd.bd_psnr(baseline_rate, baseline_m, group_rate, group_m, method='akima', min_overlap=0)
            
            print(f"{bd_rate_val:>15.4f} | {bd_psnr_val:>15.4f}")
        except Exception as e:
            print(f"{'Error':<15} | {'Error':<15}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bjontegaard analysis from CSV.')
    parser.add_argument('csv_file', help='Path to the results CSV file.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File {args.csv_file} not found.")
        sys.exit(1)
        
    analyze_bjontegaard(args.csv_file)
