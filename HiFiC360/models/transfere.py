import pandas as pd

def format_results(input_csv_path, output_csv_path):
    # Read the input CSV
    df = pd.read_csv(input_csv_path)
    
    # Calculate the average for each codec and model_idx
    averaged = df.groupby(['codec', 'model_idx'])[['bpp', 'ws_psnr', 'ws_ssim']].mean().reset_index()
    
    # Prepare the structure for the final output
    output_data = {
        'Metric': ['group', 'bpp_real', 'ws-psnr', 'ws_ssim']
    }
    
    # Populate the dictionary with the averaged values
    for _, row in averaged.iterrows():
        codec = row['codec']
        model_idx = int(row['model_idx'])
        
        # Create a unique column name for each configuration
        col_name = f"{codec}_{model_idx}"
        
        output_data[col_name] = [
            codec,              # group
            row['bpp'],         # mapped to bpp_real
            row['ws_psnr'],     # mapped to ws-psnr
            row['ws_ssim']      # mapped to ws_ssim
        ]
        
    # Create the final DataFrame and save to CSV
    out_df = pd.DataFrame(output_data)
    out_df.to_csv(output_csv_path, index=False)
    print(f"Results successfully saved to {output_csv_path}")

if __name__ == "__main__":
    format_results('results/resultados_teste_256x512.csv', 'results/resultados_convertidos.csv')