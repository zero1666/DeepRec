# -*- coding:utf-8 -*-
"""
Description: tf2.6.0

"""

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer

from tensorflow.python.ops.init_ops import Zeros, Ones

class LayerNormalization(Layer):
    def __init__(self, axis=-1, eps=1e-9, center=True, scala=True, **kwargs):
        self.axis = axis
        self.eps = eps
        self.center = center
        self.scala = scala
        super(LayerNormalization, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], 
                                     initializer=Ones(), trainable=True)
        self.beat = self.add_weight(name='beta', shape=input_shape[-1:],
                                     initializer=Zeros(), trainable=True)
        super(LayerNormalization,self).build(input_shape)
        
    def call(self, inputs):
        """
        # Batch Nornalization 零-均值规范化（z-score标准化）
        """
        mean = K.mean(inputs, axis= self.axis, keepdims=True) # 计算均值
        variance = K.mean(tf.squarre(inputs-mean), axis =-1, keepdims=True)
        std = K.sqrt(variance + self.eps) # 计算标准差
        outputs = (inputs - mean) / std
        if self.scala:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self,):
        config = {'axis': self.axis, 'eps': self.eps, 'center': self.center, 'scale': self.scale}
        base_config = super(LayerNormalization,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
