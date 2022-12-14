import tensorflow as tf

from .activation import Dice
from .core import DNN, PredictionLayer
from .interaction import CrossNet
from .normalization import LayerNormalization
from .sequence import (SequencePoolingLayer, WeightedSequenceLayer)

from .utils import NoMask, Hash, Linear, _Add, combined_dnn_input, reduce_sum, Concat

custom_objects = {'tf': tf,
                  'DNN': DNN,
                  'PredictionLayer': PredictionLayer,
                  'CrossNet': CrossNet,
                  'Dice': Dice,
                  'SequencePoolingLayer': SequencePoolingLayer,
                  'LayerNormalization': LayerNormalization,
                  'NoMask': NoMask,
                  'Hash': Hash,
                  'Linear': Linear,
                  'Concat': Concat,
                  'WeightedSequenceLayer': WeightedSequenceLayer,
                  '_Add':_Add,
		  'reduce_sum': reduce_sum
                  }
