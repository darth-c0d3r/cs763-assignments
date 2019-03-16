import numpy as np

def get_data(filename):
	with open(filename) as file:
		data = [[int(val) for val in row.split()] for row in file.readlines()]
	max_len = max([len(row) for row in data])
	data = np.array([np.array([0]*(max_len - len(row)) + row) for row in data])
	return data

