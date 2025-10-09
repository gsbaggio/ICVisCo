#!/usr/bin/env python3
"""
Análise de compressão de imagens com múltiplas métricas de distorção.
Plota gráficos scatter de BPP vs métricas de distorção para diferentes métodos de compressão.
Versão independente que não depende de imports complexos.
"""

import os
import glob
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict


def calculate_bpp_from_files(original_path, compressed_path):
    """
    Calcula BPP (bits per pixel) comparando tamanho do arquivo original PNG 
    com o arquivo comprimido .tfci
    
    Args:
        original_path: caminho para imagem original (.png)
        compressed_path: caminho para arquivo comprimido (.tfci)
    
    Returns:
        bpp: bits per pixel
    """
    # Tamanho do arquivo comprimido em bytes
    compressed_size = os.path.getsize(compressed_path)
    
    # Carrega imagem original para obter dimensões
    img = Image.open(original_path)
    width, height = img.size
    
    # Calcula BPP
    total_pixels = width * height
    bpp = (compressed_size * 8) / total_pixels
    
    return bpp


def calculate_psnr(original_path, decompressed_path):
    """
    Calcula PSNR entre imagem original e decomprimida.
    
    Args:
        original_path: caminho para imagem original
        decompressed_path: caminho para imagem decomprimida
    
    Returns:
        psnr: Peak Signal-to-Noise Ratio
    """
    # Carrega imagens
    original_img = Image.open(original_path)
    decompressed_img = Image.open(decompressed_path)
    
    # Converte RGBA para RGB se necessário
    if original_img.mode == 'RGBA':
        original_img = original_img.convert('RGB')
    if decompressed_img.mode == 'RGBA':
        decompressed_img = decompressed_img.convert('RGB')
    
    original = np.array(original_img)
    decompressed = np.array(decompressed_img)
    
    # Garante que as imagens tenham as mesmas dimensões
    if original.shape != decompressed.shape:
        raise ValueError(f"Dimensões diferentes: {original.shape} vs {decompressed.shape}")
    
    # Implementação do PSNR
    mse = np.mean(np.square(original.astype(np.float32) - decompressed.astype(np.float32)))
    if mse == 0:
        return float('inf')  # Imagens idênticas
    psnr = 20. * np.log10(255.) - 10. * np.log10(mse)
    return psnr


def calculate_mse(original_path, decompressed_path):
    """
    Calcula MSE (Mean Squared Error) entre imagem original e decomprimida.
    
    Args:
        original_path: caminho para imagem original
        decompressed_path: caminho para imagem decomprimida
    
    Returns:
        mse: Mean Squared Error
    """
    # Carrega imagens
    original_img = Image.open(original_path)
    decompressed_img = Image.open(decompressed_path)
    
    # Converte RGBA para RGB se necessário
    if original_img.mode == 'RGBA':
        original_img = original_img.convert('RGB')
    if decompressed_img.mode == 'RGBA':
        decompressed_img = decompressed_img.convert('RGB')
    
    original = np.array(original_img)
    decompressed = np.array(decompressed_img)
    
    # Garante que as imagens tenham as mesmas dimensões
    if original.shape != decompressed.shape:
        raise ValueError(f"Dimensões diferentes: {original.shape} vs {decompressed.shape}")
    
    mse = np.mean(np.square(original.astype(np.float32) - decompressed.astype(np.float32)))
    return mse


def calculate_ssim_simple(original_path, decompressed_path):
    """
    Calcula uma versão simplificada do SSIM.
    Nota: Esta é uma implementação básica. Para SSIM completo, recomenda-se usar skimage.
    
    Args:
        original_path: caminho para imagem original
        decompressed_path: caminho para imagem decomprimida
    
    Returns:
        ssim: Structural Similarity Index (aproximado)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.color import rgb2gray
        
        # Carrega imagens
        original = np.array(Image.open(original_path))
        decompressed = np.array(Image.open(decompressed_path))
        
        # Converte para escala de cinza se necessário
        if len(original.shape) == 3:
            original = rgb2gray(original)
            decompressed = rgb2gray(decompressed)
        
        return ssim(original, decompressed)
    except ImportError:
        print("Aviso: scikit-image não disponível. Pulando cálculo de SSIM.")
        return None


def analyze_compression_folder(base_path, compression_method, metrics=['psnr']):
    """
    Analisa uma pasta de compressão específica (ex: hific-hi, hific-lo, hific-mi).
    
    Args:
        base_path: caminho base para a pasta de compressão
        compression_method: nome do método de compressão
        metrics: lista de métricas a calcular ['psnr', 'mse', 'ssim']
    
    Returns:
        dict: dicionário com métricas calculadas
    """
    original_dir = os.path.join(base_path, 'original')
    compressed_dir = os.path.join(base_path, 'compressed') 
    decompressed_dir = os.path.join(base_path, 'decompressed')
    
    # Verifica se os diretórios existem
    for dir_path in [original_dir, compressed_dir, decompressed_dir]:
        if not os.path.exists(dir_path):
            print(f"Aviso: Diretório não encontrado: {dir_path}")
            return None
    
    results = {
        'method': compression_method,
        'bpp_values': [],
        'image_names': []
    }
    
    # Inicializa listas para cada métrica
    for metric in metrics:
        results[f'{metric}_values'] = []
    
    # Busca arquivos originais
    original_files = glob.glob(os.path.join(original_dir, '*'))
    original_files = [f for f in original_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Encontrados {len(original_files)} arquivos originais em {original_dir}")
    
    for original_path in original_files:
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        
        # Caminhos correspondentes
        compressed_path = os.path.join(compressed_dir, f"{base_name}.tfci")
        decompressed_path = os.path.join(decompressed_dir, f"{base_name}.png")
        
        # Verifica se todos os arquivos existem
        if not all(os.path.exists(p) for p in [original_path, compressed_path, decompressed_path]):
            print(f"Aviso: Arquivos incompletos para {base_name}")
            print(f"  Original: {os.path.exists(original_path)}")
            print(f"  Comprimido: {os.path.exists(compressed_path)}")
            print(f"  Decomprimido: {os.path.exists(decompressed_path)}")
            continue
        
        try:
            # Calcula BPP
            bpp = calculate_bpp_from_files(original_path, compressed_path)
            results['bpp_values'].append(bpp)
            results['image_names'].append(base_name)
            
            metric_values = []
            
            # Calcula métricas solicitadas
            for metric in metrics:
                if metric == 'psnr':
                    value = calculate_psnr(original_path, decompressed_path)
                    results['psnr_values'].append(value)
                    metric_values.append(f"PSNR: {value:.2f} dB")
                elif metric == 'mse':
                    value = calculate_mse(original_path, decompressed_path)
                    results['mse_values'].append(value)
                    metric_values.append(f"MSE: {value:.2f}")
                elif metric == 'ssim':
                    value = calculate_ssim_simple(original_path, decompressed_path)
                    if value is not None:
                        results['ssim_values'].append(value)
                        metric_values.append(f"SSIM: {value:.4f}")
            
            print(f"Processado: {base_name} - BPP: {bpp:.4f}, {', '.join(metric_values)}")
            
        except Exception as e:
            print(f"Erro processando {base_name}: {str(e)}")
            continue
    
    return results


def plot_compression_analysis(all_results, metrics=['psnr'], output_dir=None):
    """
    Plota gráficos de análise de compressão.
    
    Args:
        all_results: lista de dicionários com resultados
        metrics: métricas para plotar
        output_dir: diretório para salvar gráficos (opcional)
    """
    # Configuração do matplotlib
    plt.style.use('default')
    fig, axes = plt.subplots(1, len(metrics), figsize=(8*len(metrics), 6))
    if len(metrics) == 1:
        axes = [axes]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'h']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Configurações específicas por métrica (movido para fora do loop)
        if metric == 'psnr':
            ylabel = 'PSNR (dB)'
            title = 'Rate-Distortion: BPP vs PSNR'
        elif metric == 'mse':
            ylabel = 'MSE'
            title = 'Rate-Distortion: BPP vs MSE'
        elif metric == 'ssim':
            ylabel = 'SSIM'
            title = 'Rate-Distortion: BPP vs SSIM'
        else:
            ylabel = metric.upper()
            title = f'Rate-Distortion: BPP vs {metric.upper()}'
        
        for i, result in enumerate(all_results):
            if not result or not result['bpp_values']:
                continue
                
            metric_key = f'{metric}_values'
            if metric_key not in result or not result[metric_key]:
                continue
                
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            bpp_values = result['bpp_values']
            metric_values = result[metric_key]
            
            # Calcula estatísticas
            bpp_mean = np.mean(bpp_values)
            metric_mean = np.mean(metric_values)
            metric_std = np.std(metric_values)
            
            # Plot pontos individuais conectados
            sorted_indices = np.argsort(bpp_values)
            sorted_bpp = np.array(bpp_values)[sorted_indices]
            sorted_metric = np.array(metric_values)[sorted_indices]
            
            # Linha conectando os pontos
            ax.plot(sorted_bpp, sorted_metric, color=color, alpha=0.7, linewidth=1.5)
            
            # Pontos individuais
            ax.scatter(bpp_values, metric_values,
                      c=color, marker=marker, alpha=0.8, s=80, 
                      edgecolors='black', linewidth=0.5)
            
            # Ponto médio destacado
            ax.scatter(bpp_mean, metric_mean, 
                      c=color, marker=marker, s=150, 
                      edgecolors='black', linewidth=2,
                      label=f'{result["method"]} (avg: {bpp_mean:.3f} bpp, {metric_mean:.2f})')
        
        ax.set_xlabel('BPP (bits per pixel)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Melhora os limites dos eixos
        ax.margins(0.1)
    
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'compression_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {output_path}")
    
    plt.show()


def print_summary_table(all_results, metrics):
    """
    Imprime uma tabela resumo dos resultados.
    """
    print("\n" + "="*80)
    print("RESUMO DA ANÁLISE DE COMPRESSÃO")
    print("="*80)
    
    header = f"{'Método':<15} {'BPP Médio':<12} {'BPP Std':<12}"
    for metric in metrics:
        header += f" {metric.upper() + ' Médio':<12} {metric.upper() + ' Std':<12}"
    print(header)
    print("-" * len(header))
    
    for result in all_results:
        if not result['bpp_values']:
            continue
            
        bpp_mean = np.mean(result['bpp_values'])
        bpp_std = np.std(result['bpp_values'])
        
        row = f"{result['method']:<15} {bpp_mean:<12.4f} {bpp_std:<12.4f}"
        
        for metric in metrics:
            metric_key = f'{metric}_values'
            if metric_key in result and result[metric_key]:
                metric_mean = np.mean(result[metric_key])
                metric_std = np.std(result[metric_key])
                row += f" {metric_mean:<12.3f} {metric_std:<12.3f}"
            else:
                row += f" {'N/A':<12} {'N/A':<12}"
        
        print(row)
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Análise de compressão de imagens')
    parser.add_argument('--base_dir', default='files',
                       help='Diretório base contendo as pastas de métodos')
    parser.add_argument('--methods', nargs='+', default=['hific'],
                       help='Métodos de compressão a analisar')
    parser.add_argument('--metrics', nargs='+', default=['psnr'],
                       choices=['psnr', 'mse', 'ssim'],
                       help='Métricas de distorção a calcular')
    parser.add_argument('--output_dir', default='results',
                       help='Diretório para salvar resultados')
    
    args = parser.parse_args()
    
    print("Iniciando análise de compressão...")
    print(f"Diretório base: {args.base_dir}")
    print(f"Métodos: {args.methods}")
    print(f"Métricas: {args.metrics}")
    
    all_results = []
    
    # Analisa cada método especificado
    for method in args.methods:
        method_dir = os.path.join(args.base_dir, method)
        
        if not os.path.exists(method_dir):
            print(f"Aviso: Diretório não encontrado: {method_dir}")
            continue
        
        # Busca subpastas (ex: hific-hi, hific-lo, hific-mi)
        subfolders = [d for d in os.listdir(method_dir) 
                     if os.path.isdir(os.path.join(method_dir, d))]
        
        if not subfolders:
            print(f"Aviso: Nenhuma subpasta encontrada em {method_dir}")
            continue
        
        print(f"\nEncontradas subpastas para {method}: {subfolders}")
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(method_dir, subfolder)
            compression_method = f"{method}-{subfolder}"
            
            print(f"\n{'='*50}")
            print(f"Analisando: {compression_method}")
            print(f"{'='*50}")
            
            result = analyze_compression_folder(subfolder_path, compression_method, args.metrics)
            
            if result and result['bpp_values']:
                all_results.append(result)
                print(f"✓ Sucesso: {len(result['bpp_values'])} imagens processadas")
            else:
                print(f"✗ Falha: Nenhum resultado válido para {compression_method}")
    
    if all_results:
        print_summary_table(all_results, args.metrics)
        plot_compression_analysis(all_results, args.metrics, args.output_dir)
    else:
        print("\n❌ Nenhum resultado válido encontrado.")
        print("Verifique se:")
        print("  - O diretório base existe e contém as pastas corretas")
        print("  - As subpastas contêm os diretórios 'original', 'compressed', 'decompressed'")
        print("  - Os arquivos seguem a convenção de nomenclatura esperada")


if __name__ == '__main__':
    main()