import pytest
import tensorflow as tf

from ..utils import check_model, get_test_data, SAMPLE_SIZE
from deeprec.models import DCN

@pytest.mark.parametrize(
        'cross_num,hidden_size,sparse_feature_num,cross_parameterization',
        [(0, (8,), 2, 'vector'), (1, (), 1, 'vector'), (1, (8,), 3, 'vector'),
         (0, (8,), 2, 'matrix'), (1, (), 1, 'matrix'), (1, (8,), 3, 'matrix'),
        ]
)
def test_DCN(cross_num, hidden_size, sparse_feature_num, cross_parameterization):
    model_name = "DCN"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(sample_size, sparse_feature_num=sparse_feature_num,
            dense_feature_num=sparse_feature_num)

    model = DCN(feature_columns, feature_columns, cross_num=cross_num, cross_parameterization=cross_parameterization,
            dnn_hidden_units=hidden_size, dnn_dropout=0.5)
    check_model(model, model_name, x, y)

if __name__ == "__main__":
    pass
