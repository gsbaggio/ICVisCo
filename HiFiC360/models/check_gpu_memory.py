#!/usr/bin/env python3
"""
Utilitário para verificar e limpar uso de memória GPU antes de análise de compressão.
"""

import subprocess
import os
import sys

def check_gpu_memory():
    """Verifica uso atual de memória GPU"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        for i, line in enumerate(lines):
            total, used, free = map(int, line.split(', '))
            usage_percent = (used / total) * 100
            
            print(f"GPU {i}: {used}/{total} MB usado ({usage_percent:.1f}%), {free} MB livre")
            
            if usage_percent > 90:
                print(f"⚠️  GPU {i} quase cheia! Recomendo usar CPU.")
                return False
            elif usage_percent > 70:
                print(f"⚠️  GPU {i} bastante ocupada. GPU pode falhar.")
                return False
            else:
                print(f"✅ GPU {i} tem memória suficiente.")
                return True
                
    except Exception as e:
        print(f"❌ Erro verificando GPU: {e}")
        return False

def show_gpu_processes():
    """Mostra processos usando a GPU"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', 
                               '--format=csv,noheader'], 
                              capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            print("\n🔍 Processos usando GPU:")
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    pid, name, memory = line.split(', ')
                    print(f"  PID {pid}: {name} ({memory})")
        else:
            print("\n✅ Nenhum processo de computação usando GPU")
            
    except Exception as e:
        print(f"❌ Erro listando processos GPU: {e}")

def main():
    print("=== Verificação de Memória GPU ===")
    
    gpu_available = check_gpu_memory()
    show_gpu_processes()
    
    print("\n=== Recomendações ===")
    if gpu_available:
        print("✅ GPU disponível para uso. Você pode tentar --use_gpu")
        print("💡 Comando sugerido:")
        print("python compression_analysis.py --base_dir files --methods hific --metrics psnr lpips --use_gpu")
    else:
        print("⚠️  GPU ocupada. Use CPU para evitar problemas.")
        print("💡 Comando sugerido:")
        print("python compression_analysis.py --base_dir files --methods hific --metrics psnr lpips --force_cpu")

if __name__ == "__main__":
    main()