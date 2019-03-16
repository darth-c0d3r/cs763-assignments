import torch
import numpy as np
import random

def get_data(data):
	max_len = max([len(row) for row in data])
	data = np.array([np.array([0]*(max_len - len(row)) + row) for row in data])
	return data

def one_hot_encode(X):
	N, D = X.shape
	k = np.max(X)+1
	X_new =  torch.zeros((N, D, k))
	for i in range(N):
		for d in range(D):
			X_new[i, d, X[i, d]] = 1
	return X_new

def save_tensor(file_source, file_target):
	folder = "../data/"
	data = one_hot_encode(get_data(folder+file_source))
	print("saving %s ... tensor size : "%(file_source) + str(list(data.shape)))
	torch.save(data, folder+file_target)

def read_tensor(filename):
	folder = "../data/"
	data = torch.load(folder+filename)
	return data

class Dataset:

	def __init__(self, filename, batch_size, target=False):

		self.target = target
		self.data = None
		with open(filename) as file:
			self.data = [[int(val) for val in row.split()] for row in file.readlines()]

		# data = self.preprocess(data)

		self.batch_size = batch_size
		self.size = len(self.data)
		self.num_batches = int(np.ceil(self.size / self.batch_size))

	def shuffle(self):
		random.shuffle(self.data)

	def get_batch(self, batch_idx):
		if self.target:
			return torch.tensor(get_data(self.data[(batch_idx*self.batch_size):((batch_idx+1)*self.batch_size)]))
		else:
			return one_hot_encode(get_data(self.data[(batch_idx*self.batch_size):((batch_idx+1)*self.batch_size)]))

	# def preprocess(self, data):
	# 	flat_list = [item for sublist in data for item in sublist]
	# 	unq = np.unique(flat_list)
	# 	print(unq.shape)
		