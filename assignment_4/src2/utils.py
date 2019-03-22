import torch
import numpy as np
import random

def get_device(cuda = 1):
	device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu")
	print("Using Device:", device)
	return device


def get_data(data, max_len=None):
	if max_len is None:
		max_len = max([len(row) for row in data])
	data = np.array([np.array([0]*(max_len - len(row)) + row) for row in data])
	return data

def get_unique():
	f1 = "../data/train_data.txt"
	f2 =  "../data/test_data.txt"
	with open(f1, 'r') as file:
		f1 = [[int(val) for val in row.split()] for row in file.readlines()]
	with open(f2, 'r') as file:
		f2 = [[int(val) for val in row.split()] for row in file.readlines()]
	f1_data = get_data(f1)
	f2_data = get_data(f2)
	max_len = max(f1_data.shape[1], f2_data.shape[1])
	f1_data = get_data(f1, max_len)
	f2_data = get_data(f2, max_len)
	data = np.append(f1_data, f2_data, 0)
	return list(np.unique(data))

def one_hot_encode(X, unq):

	N, D = X.shape
	X_new =  torch.zeros((N, D, len(unq)))
	for i in range(N):
		for d in range(D):
			X_new[i, d, unq.index(X[i, d])] = 1
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

	def __init__(self, filename, batch_size, device, target=False):

		self.device = device
		self.unq = get_unique()
		self.target = target
		self.data = None
		with open(filename) as file:
			self.data = [[int(val) for val in row.split()] for row in file.readlines()]

		# data = self.preprocess(data)

		self.batch_size = batch_size
		self.size = len(self.data)
		self.num_batches = int(np.ceil(self.size / self.batch_size))
		self.input_dim = len(self.unq)

	def shuffle(self):
		random.shuffle(self.data)

	def get_batch(self, batch_idx):
		if self.target:
			return torch.tensor(get_data(self.data[(batch_idx*self.batch_size):((batch_idx+1)*self.batch_size)])).to(self.device)
		else:
			return one_hot_encode(get_data(self.data[(batch_idx*self.batch_size):((batch_idx+1)*self.batch_size)]), self.unq).to(self.device)

	def print_details(self):
		print("Dataset Size: %d"%(self.size))
		print("Batch Size: %d"%(self.batch_size))
		print("Number of Batches: %d"%(self.num_batches))
		print("Input Dim: %d"%(self.input_dim))