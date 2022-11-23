# -*- coding: utf-8 -*- 
import os
import sys
import pandas  as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

root_path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#print(root_path)
sys.path.append(root_path)

from deeprec.utils.feature_column import SparseFeat, DenseFeat, get_feature_names
from deeprec.models.multitask import PLE 




if __name__ == "__main__":
    pass
    
	

