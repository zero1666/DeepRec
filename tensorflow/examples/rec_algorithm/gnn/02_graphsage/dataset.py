from __future__ import print_function

import scipy.sparse as sp
import numpy as np
from collections import namedtuple


Data = namedtuple('Data', ['x', 'y', 'adjacency_dict',
                           'train_mask', 'val_mask', 'test_mask'])

class CoraData():
    def __init__(self, data_root="data/cora/"):

        self._data_root = data_root

        self._data = self.process_data()


    def load_data(self, dataset="cora"):

        print('Loading {} dataset...'.format(dataset))

        idx_features_labels = np.genfromtxt("{}{}.content".format(self._data_root, dataset), dtype=np.dtype(str))

        edges = np.genfromtxt("{}{}.cites".format(self._data_root, dataset), dtype=np.int32)

        return idx_features_labels, edges

    def process_data(self):
        
        print("Process data ...")

        idx_features_labels, edges = self.load_data()

        features = idx_features_labels[:, 1:-1].astype(np.float32)
        features = self.normalize_feature(features)

        y = idx_features_labels[:, -1]
        labels = self.encode_onehot(y)

        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)

        for self_idx in idx:
            edges = np.vstack((edges, [self_idx, self_idx]))

        idx_map = {j: i for i, j in enumerate(idx)}
        edge_indexs = np.array(list(map(idx_map.get, edges.flatten())), dtype=np.int32)
        edge_indexs = edge_indexs.reshape(edges.shape)
        
        adjacency = {}
        for edge in edge_indexs:
            key = edge[0].astype(np.int32)
            value = edge[1].astype(np.int32)

            target_value = np.array([])
            if key in adjacency.keys():
                target_value = adjacency[key]

            target_value = np.append(target_value, value)

            adjacency.update({key : target_value})


        train_index = np.arange(150)
        val_index = np.arange(150, 500)
        test_index = np.arange(500, 2708)

        train_mask = np.zeros(edge_indexs.shape[0], dtype = np.bool)
        val_mask = np.zeros(edge_indexs.shape[0], dtype = np.bool)
        test_mask = np.zeros(edge_indexs.shape[0], dtype = np.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

        print('Dataset has {} nodes, {} edges, {} features.'.format(features.shape[0], len(adjacency), features.shape[1]))

        return Data(x=features, y=labels, adjacency_dict=adjacency,
            train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
        return labels_onehot

    def normalize_adj(self, adjacency):

        """计算 L=D^-0.5 * (A+I) * D^-0.5"""
        adjacency += sp.eye(adjacency.shape[0])    # 增加自连接
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())

        return d_hat.dot(adjacency).dot(d_hat).tocsr().todense()

    def normalize_feature(self, features):
        
        normal_features = features / features.sum(1).reshape(-1, 1)

        return normal_features

    def data(self):
        """返回Data数据对象，包括features, labes, adjacency, train_mask, val_mask, test_mask"""
        return self._data