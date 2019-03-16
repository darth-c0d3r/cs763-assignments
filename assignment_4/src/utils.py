import numpy as np
import pandas as pd

def get_data(filename):
	with open(filename) as file:
		data = [[int(val) for val in row.split()] for row in file.readlines()]
	max_len = max([len(row) for row in data])
	data = np.array([np.array([0]*(max_len - len(row)) + row) for row in data])
	return data

def one_hot_encode(X, k):
	N, D = X.shape
	X_new =  np.zeros((N, D, k))
	for i in range(N):
		for d in range(D):
			X_new[i, d, X[i, d]] = 1
	return X_new

def hist(filename):
	fl = pd.read_csv(filename)
	his = pd.DataFrame(fl).hist(bins=10)