# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implement the components needed for HiFiC.

For more details, see the paper: https://arxiv.org/abs/2006.09965

The default values for all constructors reflect what was used in the paper.
"""

import collections
import math
from compare_gan.architectures import abstract_arch
from compare_gan.architectures import arch_ops
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

from hific.helpers import ModelMode


class SWHDCtf(tf.keras.layers.Layer):
    """
    Spherical Weighted Hybrid Dilated Convolution
    TensorFlow / Keras implementation
    """

    def __init__(self, filters, kernel_size, dilations=[1, 2, 3], padding="same", strides=1, learn_weights=True, **kwargs):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.N = len(dilations)
        self.learn_weights = learn_weights
        
        # Se 1, os pesos sao aprendidos. Se 0, usa o calculo geometrico SWHDC.
        # Voce pode mudar isso para True por padrao se quiser sempre aprender.
        
    def build(self, input_shape):
        in_channels = input_shape[-1]
        kh, kw = self.kernel_size, self.kernel_size
        self.kernel = self.add_weight(
            name="kernel",
            shape=(kh, kw, in_channels, self.filters),
            trainable=True,
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.filters,),
            trainable=True,
        )

        if self.learn_weights:
            # Perfil de pesos latentes com resolucao fixa (ex: 128 pontos latitudinais)
            # Shape: [1, 128, N] -> Onde N é o número de dilatações
            self.learned_profile = self.add_weight(
                name="learned_weights_profile",
                shape=(1, 128, self.N),
                initializer=tf.initializers.random_normal(mean=0.0, stddev=0.1),
                trainable=True
            )

        super().build(input_shape)

    def circular_pad_width(self, x, pad):
        """
        Circular padding along width (axis=2 for NHWC).
        """
        if pad == 0:
            return x

        left = x[:, :, -pad:, :]
        right = x[:, :, :pad, :]
        return tf.concat([left, x, right], axis=2)   

    def call(self, x):

        def reflect_or_symmetric_pad_height(x, v_pad, H):
            # Usar SYMMETRIC padding (repetir a borda) é mais estável nos polos 
            # do que REFLECT ou Zeros para imagens 360, evitando descontinuidades.
            return tf.pad(
                  x,
                  paddings=[[0, 0], [v_pad, v_pad], [0, 0], [0, 0]],
                  mode="SYMMETRIC",
            )

        """
        x: [B, H, W, C] (NHWC)
        """
        B, H, W, C = tf.unstack(tf.shape(x))
        
        row_wise_weights = None
        
        # Calcular H float para usos matemáticos
        H_float = tf.cast(H, tf.float32)

        if self.learn_weights:
            # === Modo Aprendido ===
            # Interpolar o perfil aprendido (128) para a altura atual (H)
            # learned_profile: [1, 128, N]
            
            # Precisamos expandir dimensão para usar resize image: [Batch, H, W, Ch]
            # Vamos tratar: Batch=1, H=128, W=N, C=1 para redimensionar a altura
            # Mas tf.image.resize redimensiona H e W.
            
            # Estratégia: [1, 128, N] -> expand -> [1, 128, N, 1]
            profile_expanded = tf.expand_dims(self.learned_profile, axis=-1)
            
            # Resize para [1, H, N, 1]
            profile_resized = tf.image.resize(
                profile_expanded, 
                size=[H, self.N],
                method=tf.image.ResizeMethod.BILINEAR
            )
            
            # Remove dimensões extras -> [H, N]
            # O profile_resized terá valores arbitrarios. Aplicamos Softmax
            # ao longo da dimensão das dilatações (axis=1 agora após reshape ou axis=2 antes)
            
            w_interpolated = tf.reshape(profile_resized, [H, self.N]) # [H, N]
            
            # Softmax garante que a soma dos pesos das dilatações seja 1
            row_wise_weights = tf.nn.softmax(w_interpolated, axis=-1) # [H, N]
            
            # Transpor para ficar [N, H] como o código original espera
            row_wise_weights = tf.transpose(row_wise_weights, [1, 0]) # [N, H]

        else:
            # === Modo Calculado (SWHDC Original) ===
            # ===== Compute Rs =====
            # Usar centro dos pixels evita singularidade exata em 0 e pi
            # phi vai de (0.5/H)*pi ate (1 - 0.5/H)*pi
            step = math.pi / H_float
            start = 0.5 * step
            phi = tf.linspace(start, math.pi - start, H)
            
            eps = tf.keras.backend.epsilon()

            # Com pixel centers, sin(phi) nunca é 0, mas mantemos eps por segurança
            Rs = tf.minimum(
                tf.cast(self.N, tf.float32),
                tf.abs(1.0 / (tf.sin(phi) + eps))
            )  # [H]

            Rs = tf.expand_dims(Rs, axis=0)  # [1, H]

            dilations_tensor = tf.cast(
                tf.reshape(self.dilations, [self.N, 1]), tf.float32
            )  # [N, 1]

            cR = tf.math.ceil(Rs)
            fR = tf.math.floor(Rs)

            mask_exact = tf.equal(dilations_tensor, Rs)
            mask_floor = tf.logical_and(tf.equal(dilations_tensor, fR), tf.logical_not(mask_exact))
            mask_ceil = tf.logical_and(
                tf.equal(dilations_tensor, cR),
                tf.logical_not(tf.logical_or(mask_exact, mask_floor))
            )

            row_wise_weights = tf.zeros((self.N, H), dtype=tf.float32)
            
            # Broadcast values to match [N, H] for tf.where
            diff_cR_Rs = tf.tile(cR - Rs, [self.N, 1])
            diff_Rs_fR = tf.tile(Rs - fR, [self.N, 1])

            row_wise_weights = tf.where(mask_exact, tf.ones_like(row_wise_weights), row_wise_weights)
            row_wise_weights = tf.where(mask_floor, diff_cR_Rs, row_wise_weights)
            row_wise_weights = tf.where(mask_ceil, diff_Rs_fR, row_wise_weights)

        outputs = []

        # ===== Loop over dilations =====
        for idx, dilation_rate in enumerate(self.dilations):
            # Padding sizes
            v_pad = (self.kernel_size - 1) // 2
            h_pad = dilation_rate * (self.kernel_size - 1) // 2

            x2 = self.circular_pad_width(x, h_pad)
            x2 = reflect_or_symmetric_pad_height(x2, v_pad, H)


            # Dilated convolution (vertical dilation = 1, horizontal dilation = dilation_rate)
            out = tf.nn.conv2d(
                x2,
                self.kernel,
                strides=[1, 1, 1, 1],
                padding="VALID",
                dilations=[1, 1, dilation_rate, 1],
            )
            out = tf.nn.bias_add(out, self.bias)

            # Apply row-wise weights
            weights = tf.reshape(row_wise_weights[idx], [1, H, 1, 1])
            out = out * weights

            outputs.append(out)

        # Sum over dilations
        outputs = tf.stack(outputs, axis=0)  # [N, B, H, W, C]
        return tf.reduce_sum(outputs, axis=0)  # [B, H, W, C]

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


# Output of discriminator, where real and fake are merged into single tensors.
DiscOutAll = collections.namedtuple(
    "DiscOutAll",
    ["d_all", "d_all_logits"])


# Split each tensor in a  DiscOutAll into 2.
DiscOutSplit = collections.namedtuple(
    "DiscOutSplit",
    ["d_real", "d_fake",
     "d_real_logits", "d_fake_logits"])


EntropyInfo = collections.namedtuple(
    "EntropyInfo",
    "noisy quantized nbits nbpp qbits qbpp",
)

FactorizedPriorInfo = collections.namedtuple(
    "FactorizedPriorInfo",
    "decoded latent_shape total_nbpp total_qbpp bitstring",
)

HyperInfo = collections.namedtuple(
    "HyperInfo",
    "decoded latent_shape hyper_latent_shape "
    "nbpp side_nbpp total_nbpp qbpp side_qbpp total_qbpp "
    "bitstream_tensors",
)


class Encoder(tf.keras.Sequential):
  """Encoder architecture."""

  def __init__(self,
               name="Encoder",
               num_down=4,
               num_filters_base=60,
               num_filters_bottleneck=220):
    """Instantiate model.

    Args:
      name: Name of the layer.
      num_down: How many downsampling layers to use.
      num_filters_base: Num filters to base multiplier on.
      num_filters_bottleneck: Num filters to output for bottleneck (latent).
    """
    self._num_down = num_down

    model = [
        SWHDCtf(
            filters=num_filters_base, kernel_size=7, dilations=[1, 2, 3]),
        ChannelNorm(),
        tf.keras.layers.ReLU()
    ]

    for i in range(num_down):
      model.extend([
          tf.keras.layers.Conv2D(
              filters=num_filters_base * 2 ** (i + 1),
              kernel_size=3, padding="same", strides=2),
          ChannelNorm(),
          tf.keras.layers.ReLU()])

    model.append(
        SWHDCtf(
            filters=num_filters_bottleneck,
            kernel_size=3, dilations=[1, 2, 3]))

    super(Encoder, self).__init__(layers=model, name=name)

  @property
  def num_downsampling_layers(self):
    return self._num_down


class Decoder(tf.keras.layers.Layer):
  """Decoder/generator architecture."""

  def __init__(self,
               name="Decoder",
               num_up=4,
               num_filters_base=60,
               num_residual_blocks=9,
              ):
    """Instantiate layer.

    Args:
      name: name of the layer.
      num_up: how many upsampling layers.
      num_filters_base: base number of filters.
      num_residual_blocks: number of residual blocks.
    """
    head = [ChannelNorm(),
            SWHDCtf(
                filters=num_filters_base * (2 ** num_up),
                kernel_size=3, dilations=[1, 2, 3]),
            ChannelNorm()]

    residual_blocks = []
    for block_idx in range(num_residual_blocks):
      residual_blocks.append(
          ResidualBlock(
              filters=num_filters_base * (2 ** num_up),
              kernel_size=3,
              name="block_{}".format(block_idx),
              activation="relu",
              padding="same"))

    tail = []
    for scale in reversed(range(num_up)):
      filters = num_filters_base * (2 ** scale)
      tail += [
          tf.keras.layers.Conv2DTranspose(
              filters=filters,
              kernel_size=3, padding="same",
              strides=2),
          ChannelNorm(),
          tf.keras.layers.ReLU()]

    # Final conv layer.
    tail.append(
        SWHDCtf(
            filters=3,
            kernel_size=7,
            dilations=[1, 2, 3]))

    self._head = tf.keras.Sequential(head)
    self._residual_blocks = tf.keras.Sequential(residual_blocks)
    self._tail = tf.keras.Sequential(tail)

    super(Decoder, self).__init__(name=name)

  def call(self, inputs):
    after_head = self._head(inputs)
    after_res = self._residual_blocks(after_head)
    after_res += after_head  # Skip connection
    return self._tail(after_res)


class ResidualBlock(tf.keras.layers.Layer):
  """Implement a residual block."""

  def __init__(
      self,
      filters,
      kernel_size,
      name=None,
      activation="relu",
      **kwargs_conv2d):
    """Instantiate layer.

    Args:
      filters: int, number of filters, passed to the conv layers.
      kernel_size: int, kernel_size, passed to the conv layers.
      name: str, name of the layer.
      activation: function or string, resolved with keras.
      **kwargs_conv2d: Additional arguments to be passed directly to Conv2D.
        E.g. 'padding'.
    """
    super(ResidualBlock, self).__init__()

    kwargs_conv2d["filters"] = filters
    kwargs_conv2d["kernel_size"] = kernel_size

    block = [
        SWHDCtf(dilations=[1, 2, 3], **kwargs_conv2d),
        ChannelNorm(),
        tf.keras.layers.Activation(activation),
        SWHDCtf(dilations=[1, 2, 3], **kwargs_conv2d),
        ChannelNorm()]

    self.block = tf.keras.Sequential(name=name, layers=block)

  def call(self, inputs, **kwargs):
    return inputs + self.block(inputs, **kwargs)


class ChannelNorm(tf.keras.layers.Layer):
  """Implement ChannelNorm.

  Based on this paper and keras' InstanceNorm layer:
    Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton.
    "Layer normalization."
    arXiv preprint arXiv:1607.06450 (2016).
  """

  def __init__(self,
               epsilon: float = 1e-3,
               center: bool = True,
               scale: bool = True,
               beta_initializer="zeros",
               gamma_initializer="ones",
               **kwargs):
    """Instantiate layer.

    Args:
      epsilon: For stability when normalizing.
      center: Whether to create and use a {beta}.
      scale: Whether to create and use a {gamma}.
      beta_initializer: Initializer for beta.
      gamma_initializer: Initializer for gamma.
      **kwargs: Passed to keras.
    """
    super(ChannelNorm, self).__init__(**kwargs)

    self.axis = -1
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = tf.keras.initializers.get(beta_initializer)
    self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)

  def build(self, input_shape):
    self._add_gamma_weight(input_shape)
    self._add_beta_weight(input_shape)
    self.built = True
    super().build(input_shape)

  def call(self, inputs, modulation=None):
    mean, variance = self._get_moments(inputs)
    # inputs = tf.Print(inputs, [mean, variance, self.beta, self.gamma], "NORM")
    return tf.nn.batch_normalization(
        inputs, mean, variance, self.beta, self.gamma, self.epsilon,
        name="normalize")

  def _get_moments(self, inputs):
    # Like tf.nn.moments but unbiased sample std. deviation.
    # Reduce over channels only.
    mean = tf.reduce_mean(inputs, [self.axis], keepdims=True, name="mean")
    variance = tf.reduce_sum(
        tf.squared_difference(inputs, tf.stop_gradient(mean)),
        [self.axis], keepdims=True, name="variance_sum")
    # Divide by N-1
    inputs_shape = tf.shape(inputs)
    counts = tf.reduce_prod([inputs_shape[ax] for ax in [self.axis]])
    variance /= (tf.cast(counts, tf.float32) - 1)
    return mean, variance

  def _add_gamma_weight(self, input_shape):
    dim = input_shape[self.axis]
    shape = (dim,)

    if self.scale:
      self.gamma = self.add_weight(
          shape=shape,
          name="gamma",
          initializer=self.gamma_initializer)
    else:
      self.gamma = None

  def _add_beta_weight(self, input_shape):
    dim = input_shape[self.axis]
    shape = (dim,)

    if self.center:
      self.beta = self.add_weight(
          shape=shape,
          name="beta",
          initializer=self.beta_initializer)
    else:
      self.beta = None


class _PatchDiscriminatorCompareGANImpl(abstract_arch.AbstractDiscriminator):
  """PatchDiscriminator architecture.

  Implemented as a compare_gan layer. This has the benefit that we can use
  spectral_norm from that framework.
  """

  def __init__(self,
               name,
               num_filters_base=64,
               num_layers=3,
               ):
    """Instantiate discriminator.

    Args:
      name: Name of the layer.
      num_filters_base: Number of base filters. will be multiplied as we
        go down in resolution.
      num_layers: Number of downscaling convolutions.
    """

    super(_PatchDiscriminatorCompareGANImpl, self).__init__(
        name, batch_norm_fn=None, layer_norm=False, spectral_norm=True)

    self._num_layers = num_layers
    self._num_filters_base = num_filters_base

  def __call__(self, x):
    """Overwriting compare_gan's __call__ as we only need `x`."""
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      return self.apply(x)

  def apply(self, x):
    """Overwriting compare_gan's apply as we only need `x`."""
    if not isinstance(x, tuple) or len(x) != 2:
      raise ValueError("Expected 2-tuple, got {}".format(x))
    x, latent = x
    x_shape = tf.shape(x)

    # Upscale and fuse latent.
    latent = arch_ops.conv2d(latent, 12, 3, 3, 1, 1,
                             name="latent", use_sn=self._spectral_norm)
    latent = arch_ops.lrelu(latent, leak=0.2)
    latent = tf.image.resize(latent, [x_shape[1], x_shape[2]],
                             tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x = tf.concat([x, latent], axis=-1)

    # The discriminator:
    k = 4
    net = arch_ops.conv2d(x, self._num_filters_base, k, k, 2, 2,
                          name="d_conv_head", use_sn=self._spectral_norm)
    net = arch_ops.lrelu(net, leak=0.2)

    num_filters = self._num_filters_base
    for i in range(self._num_layers - 1):
      num_filters = min(num_filters * 2, 512)
      net = arch_ops.conv2d(net, num_filters, k, k, 2, 2,
                            name=f"d_conv_{i}", use_sn=self._spectral_norm)
      net = arch_ops.lrelu(net, leak=0.2)

    num_filters = min(num_filters * 2, 512)
    net = arch_ops.conv2d(net, num_filters, k, k, 1, 1,
                          name="d_conv_a", use_sn=self._spectral_norm)
    net = arch_ops.lrelu(net, leak=0.2)

    # Final 1x1 conv that maps to 1 Channel
    net = arch_ops.conv2d(net, 1, k, k, 1, 1,
                          name="d_conv_b", use_sn=self._spectral_norm)

    out_logits = tf.reshape(net, [-1, 1])  # Reshape all into batch dimension.
    out = tf.nn.sigmoid(out_logits)

    return DiscOutAll(out, out_logits)


class _CompareGANLayer(tf.keras.layers.Layer):
  """Base class for wrapping compare_gan classes as keras layers.

  The main task of this class is to provide a keras-like interface, which
  includes a `trainable_variables`. This is non-trivial however, as
  compare_gan uses tf.get_variable. So we try to use the name scope to find
  these variables.
  """

  def __init__(self,
               name,
               compare_gan_cls,
               **compare_gan_kwargs):
    """Constructor.

    Args:
      name: Name of the layer. IMPORTANT: Setting this to the same string
        for two different layers will cause unexpected behavior since variables
        are found using this name.
      compare_gan_cls: A class from compare_gan, which should inherit from
        either AbstractGenerator or AbstractDiscriminator.
      **compare_gan_kwargs: keyword arguments passed to compare_gan_cls to
        construct it.
    """
    super(_CompareGANLayer, self).__init__(name=name)
    compare_gan_kwargs["name"] = name
    self._name = name
    self._model = compare_gan_cls(**compare_gan_kwargs)

  def call(self, x):
    return self._model(x)

  @property
  def trainable_variables(self):
    """Get trainable variables."""
    # Note: keras only returns something if self.training is true, but we
    # don't have training as a flag to the constructor, so we always return.
    # However, we only call trainable_variables when we are training.
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._model.name)


class Discriminator(_CompareGANLayer):

  def __init__(self):
    super(Discriminator, self).__init__(
        name="Discriminator",
        compare_gan_cls=_PatchDiscriminatorCompareGANImpl)


class Hyperprior(tf.keras.layers.Layer):
  """Hyperprior architecture (probability model)."""

  def __init__(self,
               num_chan_bottleneck=220,
               num_filters=320,
               name="Hyperprior"):
    super(Hyperprior, self).__init__(name=name)

    self._num_chan_bottleneck = num_chan_bottleneck
    self._num_filters = num_filters
    self._analysis = tf.keras.Sequential([
        tfc.SignalConv2D(
            num_filters, (3, 3), name=f"layer_{name}_0",
            corr=True,
            padding="same_zeros", use_bias=True,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            num_filters, (5, 5), name=f"layer_{name}_1",
            corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=tf.nn.relu),
        tfc.SignalConv2D(
            num_filters, (5, 5), name=f"layer_{name}_2",
            corr=True, strides_down=2,
            padding="same_zeros", use_bias=True,
            activation=None)], name="HyperAnalysis")

    def _make_synthesis(syn_name):
      return tf.keras.Sequential([
          tfc.SignalConv2D(
              num_filters, (5, 5), name=f"layer_{syn_name}_0",
              corr=False, strides_up=2,
              padding="same_zeros", use_bias=True,
              kernel_parameterizer=None,
              activation=tf.nn.relu),
          tfc.SignalConv2D(
              num_filters, (5, 5), name=f"layer_{syn_name}_1",
              corr=False, strides_up=2,
              padding="same_zeros", use_bias=True,
              kernel_parameterizer=None,
              activation=tf.nn.relu),
          tfc.SignalConv2D(
              num_chan_bottleneck, (3, 3), name=f"layer_{syn_name}_2",
              corr=False,
              padding="same_zeros", use_bias=True,
              kernel_parameterizer=None,
              activation=None),
      ], name="HyperSynthesis")

    self._synthesis_scale = _make_synthesis("scale")
    self._synthesis_mean = _make_synthesis("mean")

    self._side_entropy_model = FactorizedPriorLayer()

  @property
  def losses(self):
    return self._side_entropy_model.losses

  @property
  def updates(self):
    return self._side_entropy_model.updates

  @property
  def transform_layers(self):
    return [self._analysis, self._synthesis_scale, self._synthesis_mean]

  @property
  def entropy_layers(self):
    return [self._side_entropy_model]

  def call(self, latents, image_shape, mode: ModelMode) -> HyperInfo:
    """Apply this layer to code `latents`.

    Args:
      latents: Tensor of latent values to code.
      image_shape: The [height, width] of a reference frame.
      mode: The training, evaluation or validation mode of the model.

    Returns:
      A HyperInfo tuple.
    """
    training = (mode == ModelMode.TRAINING)
    validation = (mode == ModelMode.VALIDATION)

    latent_shape = tf.shape(latents)[1:-1]
    hyper_latents = self._analysis(latents, training=training)

    # Model hyperprior distributions and entropy encode/decode hyper-latents.
    side_info = self._side_entropy_model(
        hyper_latents, image_shape=image_shape, mode=mode, training=training)
    hyper_decoded = side_info.decoded

    scale_table = np.exp(np.linspace(
        np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))

    latent_scales = self._synthesis_scale(
        hyper_decoded, training=training)
    latent_means = self._synthesis_mean(
        tf.cast(hyper_decoded, tf.float32), training=training)

    if not (training or validation):
      latent_scales = latent_scales[:, :latent_shape[0], :latent_shape[1], :]
      latent_means = latent_means[:, :latent_shape[0], :latent_shape[1], :]

    conditional_entropy_model = tfc.GaussianConditional(
        latent_scales, scale_table, mean=latent_means,
        name="conditional_entropy_model")

    entropy_info = estimate_entropy(
        conditional_entropy_model, latents, spatial_shape=image_shape)

    compressed = None
    if training:
      latents_decoded = _ste_quantize(latents, latent_means)
    elif validation:
      latents_decoded = entropy_info.quantized
    else:
      compressed = conditional_entropy_model.compress(latents)
      latents_decoded = conditional_entropy_model.decompress(compressed)

    info = HyperInfo(
        decoded=latents_decoded,
        latent_shape=latent_shape,
        hyper_latent_shape=side_info.latent_shape,
        nbpp=entropy_info.nbpp,
        side_nbpp=side_info.total_nbpp,
        total_nbpp=entropy_info.nbpp + side_info.total_nbpp,
        qbpp=entropy_info.qbpp,
        side_qbpp=side_info.total_qbpp,
        total_qbpp=entropy_info.qbpp + side_info.total_qbpp,
        # We put everything that's needed for real arithmetic coding into
        # the bistream_tensors tuple.
        bitstream_tensors=(compressed, side_info.bitstring,
                           image_shape, latent_shape, side_info.latent_shape))

    tf.summary.scalar("bpp/total/noisy", info.total_nbpp)
    tf.summary.scalar("bpp/total/quantized", info.total_qbpp)

    tf.summary.scalar("bpp/latent/noisy", entropy_info.nbpp)
    tf.summary.scalar("bpp/latent/quantized", entropy_info.qbpp)

    tf.summary.scalar("bpp/side/noisy", side_info.total_nbpp)
    tf.summary.scalar("bpp/side/quantized", side_info.total_qbpp)

    return info


def _ste_quantize(inputs, mean):
  """Calculates quantize(inputs - mean) + mean, sets straight-through grads."""
  half = tf.constant(.5, dtype=tf.float32)
  outputs = inputs
  outputs -= mean
  # Rounding latents for the forward pass (straight-through).
  outputs = outputs + tf.stop_gradient(tf.math.floor(outputs + half) - outputs)
  outputs += mean
  return outputs


class FactorizedPriorLayer(tf.keras.layers.Layer):
  """Factorized prior to code a discrete tensor."""

  def __init__(self):
    """Instantiate layer."""
    super(FactorizedPriorLayer, self).__init__(name="FactorizedPrior")
    self._entropy_model = tfc.EntropyBottleneck(
        name="entropy_model")

  def compute_output_shape(self, input_shape):
    batch_size = input_shape[0]
    shapes = (
        input_shape,  # decoded
        [2],  # latent_shape = [height, width]
        [],  # total_nbpp
        [],  # total_qbpp
        [batch_size],  # bitstring
    )
    return tuple(tf.TensorShape(x) for x in shapes)

  @property
  def losses(self):
    return self._entropy_model.losses

  @property
  def updates(self):
    return self._entropy_model.updates

  def call(self, latents, image_shape, mode: ModelMode) -> FactorizedPriorInfo:
    """Apply this layer to code `latents`.

    Args:
      latents: Tensor of latent values to code.
      image_shape: The [height, width] of a reference frame.
      mode: The training, evaluation or validation mode of the model.

    Returns:
      A FactorizedPriorInfo tuple
    """
    training = (mode == ModelMode.TRAINING)
    validation = (mode == ModelMode.VALIDATION)
    latent_shape = tf.shape(latents)[1:-1]

    with tf.name_scope("factorized_entropy_model"):
      noisy, quantized, _, nbpp, _, qbpp = estimate_entropy(
          self._entropy_model, latents, spatial_shape=image_shape)

      compressed = None
      if training:
        latents_decoded = noisy
      elif validation:
        latents_decoded = quantized
      else:
        compressed = self._entropy_model.compress(latents)

        # Decompress using the spatial shape tensor and get tensor coming out of
        # range decoder.
        num_channels = latents.shape[-1].value
        latents_decoded = self._entropy_model.decompress(
            compressed, shape=tf.concat([latent_shape, [num_channels]], 0))

      return FactorizedPriorInfo(
          decoded=latents_decoded,
          latent_shape=latent_shape,
          total_nbpp=nbpp,
          total_qbpp=qbpp,
          bitstring=compressed)


def estimate_entropy(entropy_model, inputs, spatial_shape=None) -> EntropyInfo:
  """Compresses `inputs` with the given entropy model and estimates entropy.

  Args:
    entropy_model: An `EntropyModel` instance.
    inputs: The input tensor to be fed to the entropy model.
    spatial_shape: Shape of the input image (HxW). Must be provided for
      `valid == False`.

  Returns:
    The 'noisy' and quantized inputs, as well as differential and discrete
    entropy estimates, as an `EntropyInfo` named tuple.
  """
  # We are summing over the log likelihood tensor, so we need to explicitly
  # divide by the batch size.
  batch = tf.cast(tf.shape(inputs)[0], tf.float32)

  # Divide by this to flip sign and convert from nats to bits.
  quotient = tf.constant(-np.log(2), dtype=tf.float32)

  num_pixels = tf.cast(tf.reduce_prod(spatial_shape), tf.float32)

  # Compute noisy outputs and estimate differential entropy.
  noisy, likelihood = entropy_model(inputs, training=True)
  log_likelihood = tf.log(likelihood)
  nbits = tf.reduce_sum(log_likelihood) / (quotient * batch)
  nbpp = nbits / num_pixels

  # Compute quantized outputs and estimate discrete entropy.
  quantized, likelihood = entropy_model(inputs, training=False)
  log_likelihood = tf.log(likelihood)
  qbits = tf.reduce_sum(log_likelihood) / (quotient * batch)
  qbpp = qbits / num_pixels

  return EntropyInfo(noisy, quantized, nbits, nbpp, qbits, qbpp)
