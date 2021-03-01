import pandas as pd
import numpy as np

def NRMSE(actual, missing, imputation, cat_feature = None):
  if cat_feature != None:
    if not bool(cat_feature):
      raise ValueError("Пустой список индексов категориальных признаков")
    list_num_feature = np.delete(imputation.columns, cat_feature, 0)
  else:
    list_num_feature = imputation.columns
  
  mask = missing.isna()
  actual = actual[mask]
  imputation = imputation[mask]
  ximp = []
  xtrue = []
  for i in list_num_feature:
    ximp = ximp + imputation[i].dropna().to_list()
    xtrue = xtrue + actual[i].dropna().to_list()
  res = np.sqrt(np.mean((np.array(ximp) - np.array(xtrue) ** 2)) / np.var(np.array(xtrue)))
  return {'NRMSE': res}

def PFC(actual, missing, imputation, cat_feature = None):
  if cat_feature != None:
    if not bool(cat_feature):
      raise ValueError("Пустой список индексов категориальных признаков")
    mask = missing.isna()
    actual = actual[mask]
    imputation = imputation[mask]
    sum_na = 0
    sum_error = 0
    for i in imputation.columns[cat_feature]:
      sum_na = sum_na + mask[i].sum()
      res = 0
      for k, j in zip(imputation[i].dropna().to_list(), actual[i].dropna().to_list()):
        if (k != j):
          res = res + 1
      sum_error = sum_error + res
    return {'PFC': sum_error/sum_na}
  else:
    return np.nan
  
def mixError(ximp, xmis, xtrue, cat_feature = None):
  return [NRMSE(ximp, xmis, xtrue, cat_feature), PFC(ximp, xmis, xtrue, cat_feature)]