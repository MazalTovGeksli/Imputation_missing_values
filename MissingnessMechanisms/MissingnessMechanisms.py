def MCAR(df, per):
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