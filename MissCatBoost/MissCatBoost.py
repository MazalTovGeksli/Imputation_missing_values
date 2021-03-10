from catboost import Pool, CatBoostClassifier, CatBoostRegressor
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

class MissCatBoost():

  def __init__(self, max_iter=10, missing_values=np.nan, n_estimators=100, depth=6, criterion=('RMSE', 'MultiClass')):
    self.max_iter = max_iter
    self.missing_values = missing_values
    self.n_estimators = n_estimators
    self.depth = depth
    self.criterion = criterion
    self.iter_list = []
    self.reg_score_list = []
    self.class_score_list = []

  def _initial_missing_value(self, data, cat_vars = None):
    if cat_vars == None:
      data[data.columns] = self._initial_missing_numeric_value(data, data.columns)
    else:
      if not bool(cat_vars):
        raise ValueError("Пустой список индексов категориальных признаков")

      col_cat = data.columns[cat_vars]
      col_num = np.delete(data.columns, cat_vars, 0)

      if len(col_num) == 0:
        data[col_cat] = self._initial_missing_categorical_value(data, col_cat)
      else:
        data[col_cat] = self._initial_missing_categorical_value(data, col_cat)
        data[col_num] = self._initial_missing_numeric_value(data, col_num)

    return data

  def _initial_missing_numeric_value(self, data, col_num):
    sumple_imp = SimpleImputer(missing_values = np.nan, strategy = 'median')
    res2 = pd.DataFrame(sumple_imp.fit_transform(data[col_num]), columns = data[col_num].columns)
    return res2

  def _initial_missing_categorical_value(self, data, col_cat):
    sumple_imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
    res1 = pd.DataFrame(sumple_imp.fit_transform(data[col_cat]), columns = data[col_cat].columns)
    return res1

  def _train_test_val_split(self, data, i, mask):
    test = data[mask[i]]
    train = data.drop(test.index, axis = 0)

    y_test = test[i].reset_index(drop = True)
    y_train = train[i]
    X_test = test.drop([i], axis = 1)
    X_train = train.drop([i], axis = 1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.10)
    return y_test, y_train, y_val, X_test, X_train, X_val
  
  def _fit_predict_model(self, model, X_train, y_train, X_test, y_test, X_val, y_val, cat_features, flag):
    val_pool = Pool(X_val, y_val, cat_features = cat_features)
    if flag:
      model.fit(X_train, y_train, eval_set = val_pool, cat_features = cat_features)
      y_test = model.predict(X_test)
      y_test = np.reshape(y_test, (len(y_test)))
      #error_class = model.best_score_['validation']['MultiClass']       #class error
      error_class = 1 - accuracy_score(y_val, model.predict(X_val))
      return error_class, y_test
    else:
      model.fit(X_train, y_train, eval_set = val_pool, cat_features = cat_features)
      y_test = model.predict(X_test)
      #error_reg = model.best_score_['validation']['RMSE'] / np.std(y_val) #reg error
      #error_reg = np.sqrt(np.mean((np.array(model.predict(X_val)) - np.array(y_val)) ** 2) / np.var(np.array(y_val)))
      error_reg = 1 - (r2_score(np.array(y_val), np.array(model.predict(X_val))) if r2_score(np.array(y_val), np.array(model.predict(X_val))) > 0 else 0)
      return error_reg, y_test

  def print_loss(self):
    b = []
    for i, j in zip(self.reg_score_list, self.class_score_list):
      b.append(i + j)
    fig, ax = plt.subplots()
    # make a plot
    ax.plot(self.iter_list, b, color = "red", marker = "o")
    # set x-axis label
    ax.set_xlabel("iter", fontsize = 14)
    # set y-axis label
    ax.set_ylabel("sum(R^2, PFC)", color = "red", fontsize = 14)
    plt.show()

    # create figure and axis objects with subplots()
    fig, ax = plt.subplots()
    # make a plot
    ax.plot(self.iter_list, self.reg_score_list, color = "red", marker = "o")
    # set x-axis label
    ax.set_xlabel("iter", fontsize = 14)
    # set y-axis label
    ax.set_ylabel("R^2", color = "red",fontsize = 14)

    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(self.iter_list, self.class_score_list, color = "blue", marker = "o")
    ax2.set_ylabel("PFC", color="blue", fontsize = 14)
    plt.show()

  def _fit(self, X, cat_vars = None):
    pass

  def _transform(self, X):
    pass
  
  def fit_transform(self, data, cat_vars = None):
    X = data.copy()
    # First replace missing values with NaN if it is something else
    if self.missing_values not in ['NaN', np.nan]:
      X[np.where(X == self.missing_values)] = np.nan
    
    # Check if any column has all missing
    mask = X.isna()
    if np.any(mask.sum(axis=0) >= (X.shape[0])):
      raise ValueError("One or more columns have all rows missing.")
    
    # Создаём список имен категориальных признаков
    self.cat_vars_ = X.columns[cat_vars] if cat_vars != None else []
    
    # Identify numerical variables
    num_vars = np.setdiff1d(np.arange(X.shape[1]), cat_vars)
    self.num_vars_ = X.columns[num_vars] if len(num_vars) > 0 else []

    # Now, make initial guess for missing values
    X = self._initial_missing_value(X, cat_vars)

    model_reg = CatBoostRegressor(verbose = False, iterations = self.n_estimators, loss_function = self.criterion[0], use_best_model = True, early_stopping_rounds = 50)  # Инициализируем модели задествованные в восстановление пропущенных значений
    model_class = CatBoostClassifier(verbose = False, iterations = self.n_estimators, loss_function = self.criterion[1], use_best_model = True, early_stopping_rounds = 50) # Инициализируем модели задествованные в восстановление пропущенных значений

    err_all = np.inf # Стартовая позиция для ранней остановки, общая ошибка равна бесконечности

    buff_data = X.copy()  # Сохраняем заполненный набор данных с предыдущей итерации
    iter = 1

    while iter <= self.max_iter:
      error_class = 0
      error_reg = 0

      for i in X.columns:
        y_test, y_train, y_val, X_test, X_train, X_val = self._train_test_val_split(X, i, mask)
        list_cf = list([X_train.columns.get_loc(c) for c in self.cat_vars_ if c in X_train])

        if i in self.cat_vars_:
          c_err, y_test = self._fit_predict_model(model_class, X_train, y_train, X_test, y_test, X_val, y_val, list_cf, flag = True)
          error_class = error_class + c_err
        else:
          r_err, y_test = self._fit_predict_model(model_reg, X_train, y_train, X_test, y_test, X_val, y_val, list_cf, flag = False)
          error_reg = error_reg + r_err

        X[i][mask[i]] = y_test
      
      self.iter_list.append(iter)
      self.reg_score_list.append(error_reg)
      self.class_score_list.append(error_class)

      if (error_reg + error_class) < err_all:
        print({'past_iter': err_all, 'present_iter': (error_reg + error_class)})
        err_all = error_reg + error_class
        buff_data = X.copy()
      else:
        print('***Aborting iterations*** ', {'past_iter': err_all, 'present_iter': (error_reg + error_class)})
        return buff_data

      iter = iter + 1
    
    return X