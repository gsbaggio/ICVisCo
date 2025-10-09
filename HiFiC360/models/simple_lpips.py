#!/usr/bin/env python3
"""
Implementação simplificada de LPIPS que funciona com TensorFlow 1.x
com suporte adequado para GPU.
"""

import os
import numpy as np
import tensorflow.compat.v1 as tf


class SimpleLPIPSLoss:
    """LPIPS Loss implementação simplificada para TF 1.x com suporte GPU."""
    
    def __init__(self, weight_path, use_gpu=True):
        # Verifica se o arquivo de pesos existe
        if not os.path.exists(weight_path):
            from hific import helpers
            helpers.ensure_lpips_weights_exist(weight_path)
        
        self.use_gpu = use_gpu and tf.test.is_gpu_available()
        
        # Configuração de sessão para GPU
        self.config = tf.ConfigProto()
        if self.use_gpu:
            self.config.gpu_options.allow_growth = True
            # Para RTX 3090, pode precisar limitar memória
            self.config.gpu_options.per_process_gpu_memory_fraction = 0.8
            self.config.allow_soft_placement = True
            print("LPIPS configurado para usar GPU")
        else:
            # Configuração para CPU apenas (não tenta acessar GPU)
            self.config.allow_soft_placement = True
            self.config.log_device_placement = False
            print("LPIPS configurado para usar CPU")
        
        # Carrega o grafo LPIPS
        self.graph_def = tf.GraphDef()
        with open(weight_path, "rb") as f:
            self.graph_def.ParseFromString(f.read())
    
    def __call__(self, fake_image, real_image):
        """
        Calcula LPIPS assumindo inputs em [0, 1].
        """
        # Determina dispositivo
        device = '/gpu:0' if self.use_gpu else '/cpu:0'
        
        with tf.device(device):
            # Move inputs para [-1, 1] e formato NCHW
            def _transpose_to_nchw(x):
                return tf.transpose(x, (0, 3, 1, 2))
            
            fake_image_nchw = _transpose_to_nchw(fake_image * 2 - 1.0)
            real_image_nchw = _transpose_to_nchw(real_image * 2 - 1.0)
            
            # Importa o grafo LPIPS
            loss = tf.import_graph_def(
                self.graph_def,
                input_map={"0:0": fake_image_nchw, "1:0": real_image_nchw},
                return_elements=["Reshape_10:0"]
            )[0]
            
            return tf.reduce_mean(loss)


def test_gpu_availability():
    """Testa disponibilidade e configuração da GPU."""
    print("=== GPU Diagnostic ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"GPU available: {tf.test.is_gpu_available()}")
    
    if tf.test.is_gpu_available():
        print("GPU devices:")
        try:
            for i, device in enumerate(tf.config.experimental.list_physical_devices('GPU')):
                print(f"  GPU {i}: {device}")
        except:
            # Fallback para TF 1.x
            from tensorflow.python.client import device_lib
            devices = device_lib.list_local_devices()
            gpu_devices = [d for d in devices if d.device_type == 'GPU']
            for i, device in enumerate(gpu_devices):
                print(f"  GPU {i}: {device.name} - {device.physical_device_desc}")
    
    # Testa operação simples na GPU
    try:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        with tf.Session(config=config) as sess:
            with tf.device('/gpu:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
                result = sess.run(c)
                print("✅ GPU operation successful!")
                return True
    except Exception as e:
        print(f"❌ GPU operation failed: {e}")
        return False


def test_simple_lpips_gpu():
    """Testa a implementação simplificada com GPU."""
    print("\n=== Testing LPIPS with GPU ===")
    
    # Primeiro testa disponibilidade da GPU
    gpu_available = test_gpu_availability()
    
    # Cria imagens de teste
    img1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    # Testa com GPU se disponível
    for use_gpu in [True, False] if gpu_available else [False]:
        print(f"\n--- Testing with GPU={use_gpu} ---")
        
        tf.reset_default_graph()
        
        # Converte para tensor
        img1_tensor = tf.placeholder(tf.float32, [1, 256, 256, 3])
        img2_tensor = tf.placeholder(tf.float32, [1, 256, 256, 3])
        
        # Cria modelo LPIPS
        weight_path = "lpips_weight__net-lin_alex_v0.1.pb"
        lpips_loss = SimpleLPIPSLoss(weight_path, use_gpu=use_gpu)
        lpips_value = lpips_loss(img1_tensor, img2_tensor)
        
        # Executa
        import time
        start_time = time.time()
        
        with tf.Session(config=lpips_loss.config) as sess:
            img1_norm = img1.astype(np.float32) / 255.0
            img2_norm = img2.astype(np.float32) / 255.0
            
            result = sess.run(lpips_value, {
                img1_tensor: np.expand_dims(img1_norm, 0),
                img2_tensor: np.expand_dims(img2_norm, 0)
            })
            
            elapsed = time.time() - start_time
            device_type = "GPU" if use_gpu else "CPU"
            print(f"LPIPS value ({device_type}): {result:.6f}")
            print(f"Time elapsed: {elapsed:.3f}s")


def test_simple_lpips():
    """Testa a implementação básica."""
    print("=== Basic LPIPS Test ===")
    test_simple_lpips_gpu()


if __name__ == "__main__":
    test_simple_lpips()