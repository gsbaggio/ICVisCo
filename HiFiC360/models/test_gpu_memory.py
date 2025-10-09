#!/usr/bin/env python3
"""
Script para testar uso de GPU com configurações de memória limitada.
Para RTX 3090 com TensorFlow 1.x
"""

import os
import tensorflow as tf

def test_gpu_with_memory_limit():
    """Testa GPU com configuração de memória limitada"""
    print("🔍 Testando GPU com configuração de memória...")
    
    try:
        # Configuração conservadora de memória
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.1  # Apenas 10% da GPU
        config.allow_soft_placement = True
        config.log_device_placement = False
        
        with tf.Session(config=config) as sess:
            # Teste simples de operação na GPU
            with tf.device('/gpu:0'):
                a = tf.constant([1.0, 2.0, 3.0], name='a')
                b = tf.constant([4.0, 5.0, 6.0], name='b')
                c = tf.add(a, b, name='c')
            
            result = sess.run(c)
            print(f"✅ Teste GPU bem-sucedido! Resultado: {result}")
            return True
            
    except Exception as e:
        print(f"❌ Erro no teste GPU: {e}")
        return False

def test_cpu_fallback():
    """Testa fallback para CPU"""
    print("\n🔍 Testando fallback para CPU...")
    
    try:
        # Força CPU apenas
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        config = tf.ConfigProto(device_count={'GPU': 0})
        
        with tf.Session(config=config) as sess:
            # Teste simples de operação na CPU
            a = tf.constant([1.0, 2.0, 3.0], name='a')
            b = tf.constant([4.0, 5.0, 6.0], name='b')
            c = tf.add(a, b, name='c')
            
            result = sess.run(c)
            print(f"✅ Teste CPU bem-sucedido! Resultado: {result}")
            return True
            
    except Exception as e:
        print(f"❌ Erro no teste CPU: {e}")
        return False
    finally:
        # Restaura visibilidade da GPU
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']

def main():
    print("=== Teste de Configuração de Memória GPU ===")
    
    # Primeiro tenta GPU com memória limitada
    gpu_success = test_gpu_with_memory_limit()
    
    # Sempre testa CPU como fallback
    cpu_success = test_cpu_fallback()
    
    print("\n=== Resumo dos Testes ===")
    print(f"GPU (memória limitada): {'✅ OK' if gpu_success else '❌ FALHOU'}")
    print(f"CPU (fallback): {'✅ OK' if cpu_success else '❌ FALHOU'}")
    
    if gpu_success:
        print("\n💡 Recomendação: Use --use_gpu com configuração de memória limitada")
    else:
        print("\n💡 Recomendação: Use --force_cpu para evitar problemas de memória")

if __name__ == "__main__":
    main()