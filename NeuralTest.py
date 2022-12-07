import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv(r'C:\Users\aliaw\Downloads\train.csv\train.csv')

m, n = data.shape
data_train = data[1000:m].T
Y = data_train[1:n]
print(Y.size)
print(Y.max())
one_hot_Y = np.zeros((Y.size, Y.max() + 1))
one_hot_Y[np.arange(Y.size), Y] = 1
one_hot_Y = one_hot_Y.T
