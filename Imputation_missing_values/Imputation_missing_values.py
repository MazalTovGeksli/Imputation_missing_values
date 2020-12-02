# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 14:01:30 2020

@author: Geksli
"""

from missingpy import MissForest
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class Catman:
  method_imputation_missing_value = MissForest()
  categorical_feature_list = None
  encoding_dict = None

  def __init__(self):
    pass

  def fit(self, data):
    self.categorical_feature_list = list(data.select_dtypes(include = 'object').columns)
    self.encoding_dict = defaultdict(LabelEncoder)
    
    cat_vars = [data.columns.get_loc(c) for c in self.categorical_feature_list if c in data]
    original = data.copy()
    mask = original.isnull()

    for col in self.categorical_feature_list:
      data[col][pd.isnull(data[col])] = 'XXX'
      self.encoding_dict[col] = LabelEncoder().fit(list(data[col]))
      data[col] = self.encoding_dict[col].transform(data[col])

    data = data.where(~mask, original)
    self.method_imputation_missing_value.fit(data, cat_vars)


  def transform(self, data):
    int_label = list(data.select_dtypes(include = 'object').columns) + list(data.select_dtypes(include = 'int64').columns)
    original = data.copy()
    mask = original.isnull()

    for col in self.categorical_feature_list:
      data[col][pd.isnull(data[col])] = 'XXX'
      self.encoding_dict[col] = LabelEncoder().fit(list(data[col]))
      data[col] = self.encoding_dict[col].transform(data[col])

    data = data.where(~mask, original)
    data = pd.DataFrame(self.method_imputation_missing_value.transform(data), columns = data.columns)

    for i in int_label:
      data[i] = data[i].astype(int)

    for col in self.categorical_feature_list:
      data[col] = self.encoding_dict[col].inverse_transform(data[col])

    return(data)
      
  def fit_transform(self, data):
    self.categorical_feature_list = list(data.select_dtypes(include = 'object').columns)
    self.encoding_dict = defaultdict(LabelEncoder)
    
    int_label = list(data.select_dtypes(include = 'object').columns) + list(data.select_dtypes(include = 'int64').columns)
    cat_vars = [data.columns.get_loc(c) for c in self.categorical_feature_list if c in data]
    original = data.copy()
    mask = original.isnull()

    for col in self.categorical_feature_list:
      data[col][pd.isnull(data[col])] = 'XXX'
      self.encoding_dict[col] = LabelEncoder().fit(list(data[col]))
      data[col] = self.encoding_dict[col].transform(data[col])

    data = data.where(~mask, original)
    data = pd.DataFrame(self.method_imputation_missing_value.fit_transform(data, cat_vars), columns = data.columns)

    for i in int_label:
      data[i] = data[i].astype(int)

    for col in self.categorical_feature_list:
      data[col] = self.encoding_dict[col].inverse_transform(data[col])

    return(data)