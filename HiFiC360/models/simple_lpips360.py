#!/usr/bin/env python3
"""
LPIPS 360 loss adaptado para TensorFlow 1.x e compatível com HiFiC.
Versão simplificada do lpips_360.py que funciona com TensorFlow 1.15.x
"""

import numpy as np
import os

# Força CPU apenas para evitar problemas de cuDNN
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf


class SimpleLPIPS360Loss:
    """LPIPS 360 simplificado compatível com TensorFlow 1.x."""
    
    def __init__(self, weight_path="lpips_weight__net-lin_alex_v0.1.pb", 
                 use_gpu=False, latitude_weight_type='cosine', pole_weight=0.5):
        """
        Inicializa LPIPS 360 simplificado.
        
        Args:
            weight_path: Caminho para os pesos LPIPS
            use_gpu: Se deve usar GPU (forçado False para compatibilidade)
            latitude_weight_type: Tipo de peso por latitude ('cosine', 'linear', 'quadratic')
            pole_weight: Peso para regiões polares (0.0 a 1.0)
        """
        self.weight_path = weight_path
        self.latitude_weight_type = latitude_weight_type
        self.pole_weight = pole_weight
        
        # Força CPU para evitar problemas de cuDNN
        self.use_gpu = False
        
        # Configuração para CPU apenas
        self.config = tf.ConfigProto()
        self.config.allow_soft_placement = True
        self.config.log_device_placement = False
        print("LPIPS 360 configurado para usar CPU")
        
        # Carrega o grafo LPIPS
        self.graph_def = tf.GraphDef()
        with open(weight_path, "rb") as f:
            self.graph_def.ParseFromString(f.read())
    
    def _create_latitude_weights(self, height, width):
        """
        Cria mapa de pesos baseado na latitude para imagens 360.
        
        Args:
            height: Altura da imagem
            width: Largura da imagem
            
        Returns:
            Mapa de pesos com shape [height, width]
        """
        # Cria coordenadas de latitude de -π/2 (baixo) até π/2 (topo)
        # Em projeção equiretangular, latitude varia linearmente com altura
        y_coords = tf.linspace(-np.pi/2, np.pi/2, height)
        
        if self.latitude_weight_type == 'cosine':
            # Peso coseno - reduz distorção nos pólos naturalmente
            # Peso é máximo no equador (cos(0) = 1) e mínimo nos pólos (cos(±π/2) = 0)
            latitude_weights = tf.cos(y_coords)
            # Ajusta para garantir que pólos tenham peso mínimo de pole_weight
            latitude_weights = latitude_weights * (1.0 - self.pole_weight) + self.pole_weight
            
        elif self.latitude_weight_type == 'linear':
            # Peso linear dos pólos ao equador
            abs_lat = tf.abs(y_coords)
            max_lat = np.pi / 2
            latitude_weights = 1.0 - (abs_lat / max_lat) * (1.0 - self.pole_weight)
            
        elif self.latitude_weight_type == 'quadratic':
            # Peso quadrático - transição mais gradual
            abs_lat = tf.abs(y_coords)
            max_lat = np.pi / 2
            normalized_lat = abs_lat / max_lat
            latitude_weights = 1.0 - normalized_lat**2 * (1.0 - self.pole_weight)
            
        else:
            raise ValueError(f"Tipo de peso de latitude desconhecido: {self.latitude_weight_type}")
        
        # Expande para dimensões completas da imagem [height, width]
        latitude_weights = tf.expand_dims(latitude_weights, axis=1)
        latitude_weights = tf.tile(latitude_weights, [1, width])
        
        return latitude_weights
    
    def __call__(self, fake_image, real_image):
        """
        Calcula LPIPS 360 com pesos por latitude.
        
        Args:
            fake_image: Imagem gerada [batch, height, width, channels] em [0, 1]
            real_image: Imagem real [batch, height, width, channels] em [0, 1]
            
        Returns:
            Loss escalar ponderado por importância da latitude
        """
        # Obtém dimensões da imagem
        batch_size = tf.shape(fake_image)[0]
        height = tf.shape(fake_image)[1]
        width = tf.shape(fake_image)[2]
        
        # Converte para formato NCHW esperado pelo LPIPS
        def _transpose_to_nchw(x):
            return tf.transpose(x, (0, 3, 1, 2))
        
        # Move inputs para [-1, 1] e formato NCHW
        fake_image_nchw = _transpose_to_nchw(fake_image * 2 - 1.0)
        real_image_nchw = _transpose_to_nchw(real_image * 2 - 1.0)
        
        # Calcula LPIPS usando import_graph_def (compatível com TF 1.x)
        loss = tf.import_graph_def(
            self.graph_def,
            input_map={"0:0": fake_image_nchw, "1:0": real_image_nchw},
            return_elements=["Reshape_10:0"]
        )[0]
        
        # O LPIPS retorna um escalar por amostra no batch
        # Para aplicar pesos por latitude, usamos uma aproximação:
        # calculamos os pesos médios de latitude e aplicamos como fator
        
        latitude_weights = self._create_latitude_weights(height, width)
        avg_latitude_weight = tf.reduce_mean(latitude_weights)
        
        # Aplica o fator de peso médio por latitude
        weighted_loss = tf.reduce_mean(loss) * avg_latitude_weight
        
        return weighted_loss


def test_lpips360():
    """Testa a implementação do LPIPS 360."""
    print("\n=== Testando LPIPS 360 ===")
    
    # Cria imagens de teste (formato 360: mais largura que altura)
    height, width = 512, 1024  # Proporção típica 360
    fake_img = np.random.rand(1, height, width, 3).astype(np.float32)
    real_img = np.random.rand(1, height, width, 3).astype(np.float32)
    
    # Testa diferentes configurações
    configs = [
        ("cosine", 0.3, "Foco no equador"),
        ("linear", 0.5, "Transição linear"), 
        ("quadratic", 0.7, "Transição suave")
    ]
    
    for weight_type, pole_weight, desc in configs:
        print(f"\n--- Testando {desc} ({weight_type}, pole_weight={pole_weight}) ---")
        
        try:
            lpips360 = SimpleLPIPS360Loss(
                latitude_weight_type=weight_type,
                pole_weight=pole_weight,
                use_gpu=False  # Força CPU para teste
            )
            
            with tf.Session(config=lpips360.config) as sess:
                fake_ph = tf.placeholder(tf.float32, [None, None, None, 3])
                real_ph = tf.placeholder(tf.float32, [None, None, None, 3])
                
                loss_value = lpips360(fake_ph, real_ph)
                
                result = sess.run(loss_value, {
                    fake_ph: fake_img,
                    real_ph: real_img
                })
                
                print(f"✅ LPIPS 360 calculado: {result:.6f}")
                
        except Exception as e:
            print(f"❌ Erro no teste: {e}")


if __name__ == "__main__":
    test_lpips360()