def insert_NA(df, per):
  replaced = collections.defaultdict(set)
  ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
  random.shuffle(ix)
  to_replace = int(round(per*len(ix)))
  for row, col in ix:
    if len(replaced[row]) < df.shape[1] - 1:
      df.iloc[row, col] = np.nan
      to_replace -= 1
      replaced[row].add(col)
      if to_replace == 0:
        break
  return(df)

def NRMSE(actual, missing, imputation, cat_feature):
  mask = missing.isna()
  actual = actual[mask]
  imputation = imputation[mask]
  ximp = []
  xtrue = []
  for i in np.delete(imputation.columns, cat_feature, 0):
    ximp = ximp + imputation[i].dropna().to_list()
    xtrue = xtrue + actual[i].dropna().to_list()
  res = np.sqrt(np.mean((np.array(ximp) - np.array(xtrue)) ** 2) / np.var(np.array(xtrue)))
  return res

def PFC(actual, missing, imputation, cat_feature):
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
  return sum_error/sum_na

def mixError(ximp, xmis, xtrue, cat_feature):
  return {'NRMSE': NRMSE(ximp, xmis, xtrue, cat_feature), 'PFC': PFC(ximp, xmis, xtrue, cat_feature)}