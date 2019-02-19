import torch
import torchfile
from numpy import random

def normalize_data(data):

	data = data / torch.max(data)
	data = data - torch.mean(data)
	return data

def get_device(cuda = 1):
	device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu")
	print("Using Device:", device)
	return device

def get_data(data_folder, device):

	input_data = torchfile.load(data_folder+"data.bin")
	input_labels = torchfile.load(data_folder+"labels.bin")

	input_data = torch.tensor(input_data.reshape(input_data.shape[0],-1)).double().to(device)
	input_labels = torch.tensor(input_labels.reshape(input_labels.shape[0],-1)).long().to(device)

	return input_data, input_labels

def split_data(input_data, input_labels):
	train_data = input_data[:int(0.9 * input_data.shape[0]),:]
	train_labels = input_labels[:int(0.9 * input_labels.shape[0]),:]

	# train_data = input_data[:500,:]
	# train_labels = input_labels[:500,:]

	val_data = input_data[int(0.9 * input_data.shape[0]):,:]
	val_labels = input_labels[int(0.9 * input_labels.shape[0]):,:]

	print("Training Set Size: %d" % (train_data.shape[0]))
	print("Validation Set Size: %d" % (val_data.shape[0]))
	print("Input Vector Size: %d" % (train_data.shape[1]))

	return train_data, train_labels, val_data, val_labels

def shuffle(data, labels, device):
	a = [i for i in range(data.shape[0])]
	random.shuffle(a)

	data_sh, labels_sh = data[a], labels[a]

	# data1, labels1 = torch.zeros(data.shape).double().to(device), torch.zeros(labels.shape).long().to(device)
	# for i in range(data.shape[0]):
	# 	data1[i], labels1[i] = data[a[i]], labels[a[i]]

	return data_sh, labels_sh