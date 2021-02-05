
import collections
import random
import pandas as pd
import numpy as np

def MCAR(df, per):
  df = data.copy()
  n, p = df.shape
  NAloc = np.full((n*p), False, dtype = bool)
  randomlist = random.sample(range(0, n*p), int(n * p * per))
  NAloc[randomlist] = True
  NAloc = np.reshape(NAloc, (n, p))
  df[NAloc] = np.nan
  return(df)