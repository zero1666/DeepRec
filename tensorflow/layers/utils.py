# -*- coding:utf-8 -*-
"""
From https://github.com/shenweichen/DeepCTR/blob/ec78b9b24ef848b2d08797b5a93ac77d09360217/deepctr/layers/utils.py#L40
tensorflow v2.6
"""
import tensorflow as tf
from tensorflow.python.keras.layers import Flatten,Concatenate,Layer,Add
from tensorflow.python.ops.lookup_ops import TextFileInitializer

from tensorflow.python.ops.init_ops import Zeros, glorot_normal_initializer as glorot_normal

from tensorflow.python.keras.regularizers import L2 as l2
from tensorflow.python.ops.lookup_ops import StaticHashTable


class Nomask(Layer):
    def __init__(self, **kwargs):
        super(Nomask,self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(NoMask, self).build(input_shape)
    
    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None
    
class Hash(Layer):
    """Looks up keys in a table when setup `vocabulary_path`, which outputs the corresponding values.
    If `vocabulary_path` is not set, `Hash` will hash the input to [0,num_buckets). When `mask_zero` = True,
    input value `0` or `0.0` will be set to `0`, and other value will be set in range [1,num_buckets).
    The following snippet initializes a `Hash` with `vocabulary_path` file with the first column as keys and
    second column as values:
    * `1,emerson`
    * `2,lake`
    * `3,palmer`
    >>> hash = Hash(
    ...   num_buckets=3+1,
    ...   vocabulary_path=filename,
    ...   default_value=0)
    >>> hash(tf.constant('lake')).numpy()
    2
    >>> hash(tf.constant('lakeemerson')).numpy()
    0
    Args:
        num_buckets: An `int` that is >= 1. The number of buckets or the vocabulary size + 1
            when `vocabulary_path` is setup.
        mask_zero: default is False. The `Hash` value will hash input `0` or `0.0` to value `0` when
            the `mask_zero` is `True`. `mask_zero` is not used when `vocabulary_path` is setup.
        vocabulary_path: default `None`. The `CSV` text file path of the vocabulary hash, which contains
            two columns seperated by delimiter `comma`, the first column is the value and the second is
            the key. The key data type is `string`, the value data type is `int`. The path must
            be accessible from wherever `Hash` is initialized.
        default_value: default '0'. The default value if a key is missing in the table.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, num_buckets, mask_zero=False, vocabulary_path=None, default_value=0, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        self.vocabulary_path = vocabulary_path
        self.default_value = default_value
        if self.vocabulary_path:
            initializer = TextFileInitializer(vocabulary_path, 'string', 1, 'int64', 0, delimiter=',')
            self.hash_table = StaticHashTable(initializer, default_value=self.default_value)
        super(Hash, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(Hash, self).build(input_shape)
        
    def call(self, x, mask=None, **kwargs):
        if x.dtype != tf.string:
            zero = tf.as_string(tf.zeros([1], dtype=x.dtype))
            x = tf.as_string(x,)
        else:
            zero = tf.as_string(tf.zeros([1],  dtype='int32'))
        
        if self.vocabulary_path:
            hash_x = self.hash_table.lookup(x)
            return hash_x
        
        num_buckets = self.num_buckets if self.mask_zero else self.num_buckets-1
        hash_x = tf.strings.to_hash_bucket_fast(x, num_buckets, name=None) # weak hash
        
        if self.mask_zero:
            mask = tf.cast(tf.not_equal(x, zero), dtype ='int64')
            hash_x = (hash_x + 1) * mask
        
        return hash_x
        
    
def reduce_sun(input_tensor, axis=None, keep_dims=False, name=None):
    return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keep_dims, name = name)
