import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from catboost import Pool, CatBoostClassifier, CatBoostRegressor
import Imputation_missing_values.Metrics.Metrics as metrics

def initial_missing_value(data, cat_vars):
  col_cat = data.columns[cat_vars]
  col_num = np.delete(data.columns, cat_vars, 0)

  res1 = initial_missing_categorical_value(data, col_cat)
  res2 = initial_missing_numeric_value(data, col_num)
  data[col_cat] = res1
  data[col_num] = res2
  return data

def initial_missing_numeric_value(data, col_num):
  sumple_imp = SimpleImputer(missing_values = np.nan, strategy = 'median')
  res2 = pd.DataFrame(sumple_imp.fit_transform(data[col_num]), columns = data[col_num].columns)
  return res2

def initial_missing_categorical_value(data, col_cat):
  sumple_imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
  res1 = pd.DataFrame(sumple_imp.fit_transform(data[col_cat]), columns = data[col_cat].columns)
  return res1

def sort_missing_columns(data, mask, ascending=False):
  return data[mask.sum().sort_values(ascending=ascending).index]

"""
Parameters
----------
cat_vars : list, None (default =  None)
  Список индексов признаков, являющихся категориальными.
"""

def miss_catboost(data, actual = None, cat_vars = None, iter = 5, iterations = 50, sort = None):
  if actual == None:
    miss_data = None
  else:
    miss_data = data.copy()
  
  mask = data.isna()
  if np.any(mask.sum(axis=0) >= (data.shape[0])):
    raise ValueError("One or more columns have all rows missing.") # Check if any column has all missing

  if cat_vars != None:
    cat_vars_names = data.columns[cat_vars]
  else:
    cat_vars_names = []
  
  if sort != None:                                                              # сортировка пока что не понятно зачем нужна. С ней больше мороки
    data = sort_missing_columns(data, mask, ascending=sort)
    cat_vars = [data.columns.get_loc(c) for c in cat_vars_names if c in data]
    cat_vars_names = data.columns[cat_vars]
  
  data = initial_missing_value(data, cat_vars)

  model_reg = CatBoostRegressor(verbose = False, iterations= iterations)
  model_class = CatBoostClassifier(verbose = False, iterations= iterations, loss_function = 'MultiClass')

  print(cat_vars_names)

  err_all = np.inf
  while iter > 0:
    error_class = 0
    error_reg = 0

    for i in data.columns:
      test = data[mask[i]]
      train = data.drop(test.index, axis = 0)

      y_test = test[i].reset_index(drop = True)
      y_train = train[i]
      X_test = test.drop([i], axis = 1)
      X_train = train.drop([i], axis = 1)

      list_cf = list([X_train.columns.get_loc(c) for c in cat_vars_names if c in X_train])
      if i in cat_vars_names:
        model_class.fit(X_train, y_train, cat_features = list_cf)
        y_test = model_class.predict(X_test)
        y_test = np.reshape(y_test, (len(y_test)))

        error_class = error_class + model_class.evals_result_['learn']['MultiClass'][iterations - 1]
      else:
        model_reg.fit(X_train, y_train, cat_features = list_cf)
        y_test = model_reg.predict(X_test)

        error_reg = error_reg + model_reg.evals_result_['learn']['RMSE'][iterations - 1] / np.std(y_test)

      data[i][mask[i]] = y_test

    if (error_reg + error_class)/2 < err_all:
      print({'err_all': err_all, 'sum(error_reg + error_class)': (error_reg + error_class)/2})
      err_all = (error_reg + error_class)/2
    else:
      print('ПРЕРЫВАНИЕ ЦИКЛА')
      print({'err_all': err_all, 'sum(error_reg + error_class)': (error_reg + error_class)/2})
      break

    iter = iter - 1

  return(data)