#!/usr/bin/env python3
"""
Script para diagnosticar problemas de GPU com TensorFlow 1.x e RTX 3090.
"""

import os
import subprocess
import sys


def check_nvidia_drivers():
    """Verifica drivers NVIDIA."""
    print("=== NVIDIA Drivers ===")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi funcionando:")
            print(result.stdout)
        else:
            print("❌ nvidia-smi falhou")
            print(result.stderr)
    except FileNotFoundError:
        print("❌ nvidia-smi não encontrado")


def check_cuda_installation():
    """Verifica instalação do CUDA."""
    print("\n=== CUDA Installation ===")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA Compiler encontrado:")
            print(result.stdout)
        else:
            print("❌ nvcc falhou")
    except FileNotFoundError:
        print("❌ nvcc não encontrado")
    
    # Verifica bibliotecas CUDA
    cuda_paths = [
        '/usr/local/cuda/lib64',
        '/usr/lib/x86_64-linux-gnu'
    ]
    
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"✅ Diretório CUDA encontrado: {path}")
            libcuda = os.path.join(path, 'libcuda.so.1')
            libcudart = os.path.join(path, 'libcudart.so.10.0')
            if os.path.exists(libcuda):
                print(f"  ✅ libcuda.so.1 encontrada")
            if os.path.exists(libcudart):
                print(f"  ✅ libcudart.so.10.0 encontrada")


def check_tensorflow_gpu():
    """Verifica TensorFlow e suporte a GPU."""
    print("\n=== TensorFlow GPU Support ===")
    
    try:
        import tensorflow.compat.v1 as tf
        print(f"✅ TensorFlow importado: versão {tf.__version__}")
        
        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"GPU available: {tf.test.is_gpu_available()}")
        
        if tf.test.is_gpu_available():
            print("GPU devices detectados:")
            try:
                # Tenta método TF 2.x primeiro
                devices = tf.config.experimental.list_physical_devices('GPU')
                for i, device in enumerate(devices):
                    print(f"  GPU {i}: {device}")
            except:
                # Fallback para TF 1.x
                try:
                    from tensorflow.python.client import device_lib
                    devices = device_lib.list_local_devices()
                    gpu_devices = [d for d in devices if d.device_type == 'GPU']
                    for i, device in enumerate(gpu_devices):
                        print(f"  GPU {i}: {device.name}")
                        print(f"    Memory: {device.memory_limit}")
                        print(f"    Description: {device.physical_device_desc}")
                except Exception as e:
                    print(f"  ❌ Erro listando dispositivos: {e}")
        else:
            print("❌ Nenhuma GPU detectada pelo TensorFlow")
            
    except ImportError as e:
        print(f"❌ Erro importando TensorFlow: {e}")


def test_simple_gpu_operation():
    """Testa operação simples na GPU."""
    print("\n=== Testing Simple GPU Operation ===")
    
    try:
        import tensorflow.compat.v1 as tf
        
        # Configuração conservadora para GPU
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.allow_soft_placement = True
        
        with tf.Session(config=config) as sess:
            print("Tentando operação na GPU...")
            
            with tf.device('/gpu:0'):
                # Operação simples
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[2.0], [3.0]])
                c = tf.matmul(a, b)
                
                result = sess.run(c)
                print("✅ Operação GPU bem-sucedida!")
                print(f"Resultado: {result}")
                return True
                
    except Exception as e:
        print(f"❌ Operação GPU falhou: {e}")
        return False


def check_memory_usage():
    """Verifica uso de memória da GPU."""
    print("\n=== GPU Memory Usage ===")
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                total, used, free = line.split(', ')
                print(f"GPU {i}: Total={total}MB, Used={used}MB, Free={free}MB")
                
                # Aviso se pouca memória livre
                if int(free) < 2000:  # Menos de 2GB livre
                    print(f"  ⚠️  Pouca memória livre na GPU {i}")
        else:
            print("❌ Erro verificando memória GPU")
    except Exception as e:
        print(f"❌ Erro: {e}")


def suggest_fixes():
    """Sugere correções para problemas comuns."""
    print("\n=== Sugestões de Correção ===")
    
    print("1. Para RTX 3090 com TensorFlow 1.x:")
    print("   - Instale CUDA 10.0 ou 10.1")
    print("   - Instale cuDNN 7.6.x compatível")
    print("   - Use tensorflow-gpu==1.15.0")
    
    print("\n2. Se GPU não é detectada:")
    print("   - Verifique drivers NVIDIA: sudo apt update && sudo apt install nvidia-driver-470")
    print("   - Reinicie o sistema")
    print("   - Verifique se CUDA está no PATH")
    
    print("\n3. Se há erro de memória:")
    print("   - Feche outros processos que usam GPU")
    print("   - Use config.gpu_options.allow_growth = True")
    print("   - Limite memória: config.gpu_options.per_process_gpu_memory_fraction = 0.7")
    
    print("\n4. Variáveis de ambiente úteis:")
    print("   export CUDA_VISIBLE_DEVICES=0")
    print("   export TF_FORCE_GPU_ALLOW_GROWTH=true")
    print("   export CUDA_CACHE_DISABLE=1")


def main():
    """Executa diagnóstico completo."""
    print("🔍 Diagnóstico de GPU para TensorFlow\n")
    
    check_nvidia_drivers()
    check_cuda_installation()
    check_tensorflow_gpu()
    check_memory_usage()
    
    print("\n" + "="*50)
    gpu_works = test_simple_gpu_operation()
    
    if not gpu_works:
        suggest_fixes()
    else:
        print("\n🎉 GPU funcionando corretamente!")
        print("\nPara usar no script de compressão:")
        print("python compression_analysis.py --base_dir files --methods hific --metrics psnr lpips --use_gpu")


if __name__ == "__main__":
    main()