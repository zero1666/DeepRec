# -*- coding:utf-8 -*-

import pytest
import tensorflow as tf

from ..utils_mtl import get_mtl_test_data, check_mtl_model
from ...models.multitask import PLE



def test_importpath():
    import os
    
    
@pytest.mark.parametrize(
    'num_levels, gate_dnn_hidden_units',
    [(2,()),
     (1,(4,))]
)
def test_PLE(num_levels, gate_dnn_hidden_units):
    module_name = 'PLE'
    x, y_list, dnn_feature_columns = get_mtl_test_data()
    
    model = PLE(dnn_feature_columns, num_levels = num_levels, expert_dnn_hidden_units=(8,), tower_dnn_hidden_units=(8,),
               gate_dnn_hidden_units=gate_dnn_hidden_units,
               task_types=['binary', 'binary'], task_names=['income', 'marital'])
    check_mtl_model(model, module_name, x, y_list, task_types=['binary', 'binary'])
    
    


if __name__ == "__main__":
    
    pass

    