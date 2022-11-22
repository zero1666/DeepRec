# -*- coding:utf-8 -*-
"""
Author:
    
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout
from tensorflow.python.ops.init_ops import Zeros, glorot_normal_initializer as glorot_normal
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from .activation import activation_layer



class DNN(Layer):
    """The Multi Layer Percetron
      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.
      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.
      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.
        - **activation**: Activation function to use.
        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.
        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.
        - **use_bn**: bool. Whether use BatchNormalization before activation or not.
        - **output_activation**: Activation function to use in the last layer.If ``None``,it will be same as ``activation``.
        - **seed**: A Python integer to use as random seed.
    """
    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, user_bn=False, output_activation=None,
                seed=1024, **kwargs):
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.output_activation = output_activation
        self.seed = seed
        
        super(DNN, self).__init__(**kwargs)
    
    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [ self.add_weight( name= 'kernel_' + str(i),
                                         shape=(
                                             hidden_units[i], hidden_units[i+1]),
                                         initializer = glorot_normal(seed=self.seed),
                                         regularizer=l2(self.l2_reg),
                                         trainable=True)for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias_'+str(i), shape=(self.hidden_units[i],), initializer=Zeros(), trainable=True) for i in range(len(self.hidden_units))]

        if self.user_bn:
            self.bn_layers = [BatchNormalization() for _ in range(self.hidden_units)]
        
        self.dropout_layers = [Dropout(self.dropout_rate, seed=self.seed+1) for i in range(len(self.hidden_units))]
        self.activation_layers = [activation_lyaer(self.activation) for _ in range(len(self.hidden_units))]
        
        if self.output_activation:
            self.activation_layers[-1] = activation_layer(self.output_activation)
        
        super(DNN).build(input_shape)
    
    def call(self, inputs, training=None, **kwargs):
        
        deep_inputs = inputs
        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(deep_inputs, slef.kernels[i], axes=(-1,0)), self.bias[i])
            
            if self.user_bn:
                fc = self.bn_layers[i](fc, training=training)
                
            fc = self.activation_layers[i](fc, training=True)
            
            fc = self.dropout_layers[i](fc, training=training)
            deep_inputs = fc
            
        return deep_inputs
        
    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape
            
        return tuple(shape)
    
    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate,
                  'output_activation': self.output_activation, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class PredictionLayer(Layer):
    def __init__(self, task='binary', use_bias = False, **kwargs):
        if task not in ['binary', 'multicalss', 'regression']:
            raise ValueError("task must be binary,multiclass or regression")
        self.task = task
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        if self.use_bias :
            self.gloal_bias = self.add_weight(name = 'global_bias', shape=(1,), initializer=Zeros())
        
        super(PredictionLayer, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        if self.task == 'binary':
            x = tf.sigmoid(x)
            
        output = tf.reshape(x, (-1,1))
        return output
    
    def compute_output_shape(self, input_shape):
        return (None, 1)
    
    def get_config(self,):
        config = {'task':sel.task, 'use_bias':self.use_bias}
        base_config = super(PredictionLayer,self).get_config()
        return dict(list(base_config.items()) + liset(config.items()))