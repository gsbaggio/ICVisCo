#!/usr/bin/env python3
"""
Exemplo de uso do script de análise de compressão.
Demonstra como usar o script para diferentes cenários.
"""

import os
import subprocess
import sys

def run_analysis(base_dir, methods, metrics, output_dir="results"):
    """
    Executa análise de compressão com os parâmetros especificados.
    """
    cmd = [
        sys.executable, 
        "compression_analysis_standalone.py",
        "--base_dir", base_dir,
        "--methods"] + methods + [
        "--metrics"] + metrics + [
        "--output_dir", output_dir
    ]
    
    print(f"Executando: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0


def main():
    print("=== EXEMPLOS DE USO DO SCRIPT DE ANÁLISE DE COMPRESSÃO ===\n")
    
    # Exemplo 1: Análise básica com PSNR
    print("1. Análise básica do HiFiC com PSNR:")
    success = run_analysis("files", ["hific"], ["psnr"], "results/example1")
    print(f"Sucesso: {success}\n")
    
    # Exemplo 2: Múltiplas métricas
    print("2. Análise do HiFiC com múltiplas métricas:")
    success = run_analysis("files", ["hific"], ["psnr", "mse"], "results/example2")
    print(f"Sucesso: {success}\n")
    
    # Exemplo 3: Como adicionar novos métodos no futuro
    print("3. Preparação para novos métodos (exemplo):")
    print("   Para adicionar um novo método de compressão (ex: 'jpeg2000'):")
    print("   1. Crie a estrutura de diretórios:")
    print("      files/jpeg2000/jpeg2000-high/")
    print("      files/jpeg2000/jpeg2000-medium/")
    print("      files/jpeg2000/jpeg2000-low/")
    print("   2. Cada subpasta deve conter:")
    print("      - original/: imagens originais (.png, .jpg)")
    print("      - compressed/: arquivos comprimidos (.tfci ou outro formato)")
    print("      - decompressed/: imagens decomprimidas (.png)")
    print("   3. Execute: python compression_analysis_standalone.py --methods hific jpeg2000 --metrics psnr")
    print()
    
    # Exemplo 4: Análise com SSIM (se scikit-image estiver disponível)
    print("4. Tentativa de análise com SSIM:")
    try:
        success = run_analysis("files", ["hific"], ["psnr", "ssim"], "results/example4")
        print(f"Sucesso: {success}")
    except Exception as e:
        print(f"Erro (esperado se scikit-image não estiver instalado): {e}")
    print()
    
    print("=== ESTRUTURA DE DIRETÓRIOS ESPERADA ===")
    print("""
    files/
    ├── hific/
    │   ├── hific-hi/
    │   │   ├── original/
    │   │   │   ├── image1.png
    │   │   │   └── image2.jpg
    │   │   ├── compressed/
    │   │   │   ├── image1.tfci
    │   │   │   └── image2.tfci
    │   │   └── decompressed/
    │   │       ├── image1.png
    │   │       └── image2.png
    │   ├── hific-lo/
    │   │   └── ... (mesma estrutura)
    │   └── hific-mi/
    │       └── ... (mesma estrutura)
    └── outro_metodo/  # Para métodos futuros
        ├── outro_metodo-config1/
        │   └── ... (mesma estrutura)
        └── outro_metodo-config2/
            └── ... (mesma estrutura)
    """)
    
    print("=== COMO INTERPRETAR OS RESULTADOS ===")
    print("""
    BPP (Bits Per Pixel): Taxa de compressão
    - Menor BPP = maior compressão
    - Melhor para eficiência de armazenamento/transmissão
    
    PSNR (Peak Signal-to-Noise Ratio): Qualidade da imagem
    - Maior PSNR = melhor qualidade
    - Valores típicos: 20-50 dB
    
    MSE (Mean Squared Error): Erro médio quadrático
    - Menor MSE = melhor qualidade
    - Complementar ao PSNR
    
    SSIM (Structural Similarity Index): Similaridade estrutural
    - Valores entre 0 e 1
    - Maior SSIM = melhor qualidade perceptual
    
    Gráfico Rate-Distortion:
    - Eixo X: BPP (taxa de compressão)
    - Eixo Y: Métrica de qualidade
    - Pontos conectados mostram trade-off compressão vs qualidade
    - Idealmente: baixo BPP com alta qualidade
    """)


if __name__ == "__main__":
    main()