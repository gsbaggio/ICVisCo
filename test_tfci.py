import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import sys
import os
sys.path.append(os.path.abspath('compression/models'))
import tfci

input_image = tf.placeholder(tf.uint8, [1, 256, 512, 3])
sender = tfci.instantiate_model_signature('hific-lo', 'sender')
receiver = tfci.instantiate_model_signature('hific-lo', 'receiver')
tensors = sender(input_image)
output_image, = receiver(*tensors)
print("Tensors:", tensors)
print("Output Image:", output_image)
