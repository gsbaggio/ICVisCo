"""
Patch para compatibilidade do compare_gan e tensorflow_compression com TensorFlow 1.x/2.x.

Este módulo deve ser importado antes de qualquer importação do compare_gan
para garantir que tf.AUTO_REUSE e tensorflow.contrib estejam disponíveis.
"""

import sys
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_compression as tfc
from tensorflow_compression.python.distributions import uniform_noise

# Adiciona tf.AUTO_REUSE se não existir (necessário para TensorFlow 2.x)
if not hasattr(tf, 'AUTO_REUSE'):
    tf.AUTO_REUSE = tf1.AUTO_REUSE
    print("Patch aplicado: tf.AUTO_REUSE configurado")

# Adiciona truncated_normal_initializer se não existir
if not hasattr(tf, 'truncated_normal_initializer'):
    tf.truncated_normal_initializer = tf1.truncated_normal_initializer
    print("Patch aplicado: tf.truncated_normal_initializer configurado")

# Adiciona outros inicializadores comuns que podem estar faltando
if not hasattr(tf, 'glorot_uniform_initializer'):
    tf.glorot_uniform_initializer = tf1.glorot_uniform_initializer
if not hasattr(tf, 'glorot_normal_initializer'):
    tf.glorot_normal_initializer = tf1.glorot_normal_initializer
if not hasattr(tf, 'orthogonal_initializer'):
    tf.orthogonal_initializer = tf1.orthogonal_initializer
if not hasattr(tf, 'variance_scaling_initializer'):
    tf.variance_scaling_initializer = tf1.variance_scaling_initializer

# Cria um módulo fake para tensorflow.contrib
if not hasattr(tf, 'contrib'):
    # Cria estrutura de módulos para tensorflow.contrib.tpu.python.tpu
    class FakeTPUFunction:
        """Stub para tpu_function quando TPU não está disponível."""
        pass
    
    class FakeTPUPython:
        """Stub para tpu.python."""
        tpu = type('tpu', (), {'tpu_function': FakeTPUFunction})()
    
    class FakeTPU:
        """Stub para contrib.tpu."""
        python = FakeTPUPython()
    
    class FakeContrib:
        """Stub para tensorflow.contrib."""
        tpu = FakeTPU()
    
    # Adiciona contrib ao módulo tensorflow
    tf.contrib = FakeContrib()
    
    # Também adiciona ao sys.modules para imports diretos
    sys.modules['tensorflow.contrib'] = FakeContrib()
    sys.modules['tensorflow.contrib.tpu'] = FakeTPU()
    sys.modules['tensorflow.contrib.tpu.python'] = FakeTPUPython()
    sys.modules['tensorflow.contrib.tpu.python.tpu'] = FakeTPUPython.tpu
    
    print("Patch aplicado: tensorflow.contrib configurado (TPU desabilitado)")

# Adiciona EntropyBottleneck como um wrapper para ContinuousBatchedEntropyModel
if not hasattr(tfc, 'EntropyBottleneck'):
    class EntropyBottleneck(tf.Module):
        """Wrapper para compatibilidade com a API antiga do EntropyBottleneck."""
        
        def __init__(self, name="entropy_bottleneck", **kwargs):
            super().__init__(name=name)
            # Cria uma distribuição uniforme com ruído para o modelo
            prior = uniform_noise.NoisyNormal(loc=0., scale=1.)
            # coding_rank=3 para compatibilidade com (batch, height, width, channels)
            self._model = tfc.ContinuousBatchedEntropyModel(
                prior, coding_rank=3, compression=False, **kwargs)
        
        @property
        def losses(self):
            """Retorna as perdas do modelo interno."""
            return self._model.losses if hasattr(self._model, 'losses') else []
        
        @property
        def updates(self):
            """Retorna as atualizações do modelo interno."""
            return self._model.updates if hasattr(self._model, 'updates') else []
        
        def __call__(self, bottleneck, training=True):
            """Mantém compatibilidade com a API antiga retornando (perturbed, likelihood)."""
            import numpy as np
            
            # Chama o método do modelo interno que retorna (perturbed, bits)
            bottleneck_perturbed, bits = self._model(bottleneck, training=training)
            
            # Converte bits de volta para log-likelihood (likelihood = 2^(-bits))
            # A API antiga esperava likelihood, então precisamos converter
            # bits = -log2(likelihood) => likelihood = 2^(-bits)
            log2 = tf.constant(np.log(2), dtype=bits.dtype)
            log_likelihood = -bits * log2  # converte de bits para nats
            likelihood = tf.exp(log_likelihood)
            
            return bottleneck_perturbed, likelihood
        
        def compress(self, bottleneck):
            """Comprime o bottleneck."""
            return self._model.compress(bottleneck)
        
        def decompress(self, strings, shape):
            """Descomprime as strings."""
            return self._model.decompress(strings, shape)
    
    tfc.EntropyBottleneck = EntropyBottleneck
    print("Patch aplicado: tfc.EntropyBottleneck configurado (usando ContinuousBatchedEntropyModel)")

# Adiciona GaussianConditional como um wrapper para LocationScaleIndexedEntropyModel
if not hasattr(tfc, 'GaussianConditional'):
    class GaussianConditional(tf.Module):
        """Wrapper para compatibilidade com a API antiga do GaussianConditional."""
        
        def __init__(self, scale, scale_table, mean=None, name="gaussian_conditional", **kwargs):
            super().__init__(name=name)
            import numpy as np
            
            # Armazena os parâmetros
            self._scale = scale
            self._scale_table = scale_table
            self._mean = mean if mean is not None else tf.zeros_like(scale)
            
            # Cria o modelo de entropia com indexação por escala
            # scale_table é usado para quantizar os índices de escala
            self._model = tfc.LocationScaleIndexedEntropyModel(
                tfc.NoisyNormal,
                num_scales=len(scale_table) if isinstance(scale_table, (list, np.ndarray)) else scale_table.shape[0],
                scale_fn=lambda x: x,  # função identidade
                coding_rank=3,
                compression=False,
                **kwargs
            )
            self._scale_table = tf.constant(scale_table, dtype=tf.float32) if not isinstance(scale_table, tf.Tensor) else scale_table
        
        def __call__(self, bottleneck, training=True):
            """Mantém compatibilidade com a API antiga."""
            import numpy as np
            
            # Calcula os índices de escala
            # Encontra o índice mais próximo na tabela de escalas
            scale_indexes = self._compute_scale_indexes(self._scale)
            
            # Chama o modelo com os índices
            bottleneck_perturbed, bits = self._model(
                bottleneck, scale_indexes, loc=self._mean, training=training)
            
            # Converte bits para likelihood (mesma lógica do EntropyBottleneck)
            log2 = tf.constant(np.log(2), dtype=bits.dtype)
            log_likelihood = -bits * log2
            likelihood = tf.exp(log_likelihood)
            
            return bottleneck_perturbed, likelihood
        
        def _compute_scale_indexes(self, scale):
            """Calcula os índices na tabela de escalas."""
            # Encontra o índice mais próximo para cada valor de escala
            # scale_table: [num_scales], scale: [batch, height, width, channels]
            scale = tf.maximum(scale, 1e-9)  # evita log(0)
            scale_table_log = tf.math.log(self._scale_table)
            scale_log = tf.math.log(scale)
            
            # Broadcast e encontra o índice mais próximo
            scale_log_expanded = tf.expand_dims(scale_log, -1)  # [..., 1]
            scale_table_log_expanded = tf.reshape(scale_table_log, [1, 1, 1, 1, -1])  # [1, 1, 1, 1, num_scales]
            
            # Calcula a distância
            distances = tf.abs(scale_log_expanded - scale_table_log_expanded)
            indexes = tf.argmin(distances, axis=-1, output_type=tf.int32)
            
            return indexes
        
        def compress(self, bottleneck):
            """Comprime o bottleneck."""
            scale_indexes = self._compute_scale_indexes(self._scale)
            return self._model.compress(bottleneck, scale_indexes, loc=self._mean)
        
        def decompress(self, strings):
            """Descomprime as strings."""
            scale_indexes = self._compute_scale_indexes(self._scale)
            return self._model.decompress(strings, scale_indexes, loc=self._mean)
    
    tfc.GaussianConditional = GaussianConditional
    print("Patch aplicado: tfc.GaussianConditional configurado (usando LocationScaleIndexedEntropyModel)")

# Patch para tf.summary.image para evitar erro de bad_color no TF1
# O problema é que image_summary tem um bug com o atributo bad_color em algumas versões
# Vamos substituir completamente a função para usar uma implementação segura
from tensorflow.python.summary import summary as summary_module

_original_image = summary_module.image

def _safe_image_summary(tag, tensor, max_outputs=3, collections=None, family=None, name=None):
    """Versão segura de image summary que evita o erro bad_color."""
    # Simplesmente retorna um no-op - os summaries de imagem não são essenciais para o treinamento
    # Se quiser ver as imagens, use TensorBoard após o treinamento
    with tf1.name_scope(name, "ImageSummary", [tensor]) as scope:
        # Retorna um tensor vazio como placeholder
        return tf.constant("", dtype=tf.string, name=scope)

# Substitui no módulo
summary_module.image = _safe_image_summary
tf.summary.image = _safe_image_summary
tf1.summary.image = _safe_image_summary

print("Patch aplicado: tf.summary.image desabilitado (image summaries não disponíveis)")

print("✓ Todos os patches de compatibilidade TensorFlow aplicados com sucesso")
