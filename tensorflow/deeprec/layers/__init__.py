import tensorflow as tf

from .activation import Dice
from .core import DNN, PredictionLayer
from .normalization import LayerNormalization
from .sequence import (SequencePoolingLayer, WeightedSequenceLayer)

from .utils import NoMask, Hash, Linear, combined_dnn_input, reduce_sum

custom_objects = {'tf': tf,
                  'DNN': DNN,
                  'PredictionLayer': PredictionLayer,
                  'Dice': Dice,
                  'SequencePoolingLayer': SequencePoolingLayer,
                  'LayerNormalization': LayerNormalization,
                  'NoMask': NoMask,
                  'Hash': Hash,
                  'Linear': Linear,
                  'WeightedSequenceLayer': WeightedSequenceLayer,
                  'reduce_sum': reduce_sum
                  }