import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torchvision # for data
import model
import numpy as np
from util import *
import math

# hyper-parameters
batch_size = 500
epochs = 50
fc = [1024]
size_output = 6
size = 108*108

# return normalized dataset divided into two sets
def prepare_db():
	train_dataset = torchvision.datasets.MNIST('../src/data/mnist', train=True, download=True,
											   transform=torchvision.transforms.Compose([
												   torchvision.transforms.ToTensor(),
											   ]))

	eval_dataset = torchvision.datasets.MNIST('../src/data/mnist', train=False, download=True,
											   transform=torchvision.transforms.Compose([
												   torchvision.transforms.ToTensor(),
											   ]))
	return {'train':train_dataset,'eval':eval_dataset}

device = get_device(1)
model = model.fc_net(size, fc, size_output).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train(model, optim, train_data, train_labels, val_data, val_labels):

	for epoch in range(1, epochs+1):

		loss_total = float(0)
		num_samples = 0
		num_correct = 0

		# Update (Train)
		model.train()
		for i in range(int(math.ceil(train_data.shape[0]/batch_size))):

			data = train_data[batch_size*(i) : batch_size*(i+1), :]
			target = train_labels[batch_size*(i) : batch_size*(i+1), :].reshape((-1))

			optimizer.zero_grad()
			output = model(data)
			loss = criterion(output,target.long())

			pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
			correct = pred.eq(target.data.view_as(pred).long()).cpu().sum()
			loss.backward()
			optimizer.step()

			loss_total += loss
			num_correct += correct
			num_samples += data.shape[0]

		print("[T] Epoch = %d : Loss = %f : Accuracy = %d/%d (%f)" % (epoch, loss_total/num_samples, num_correct, 
				num_samples, float(num_correct) * 100.0 / num_samples ))

		output = model(val_data)
		loss = criterion(output,val_labels.reshape(-1))
		pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
		correct = pred.eq(val_labels.data.view_as(pred).long()).cpu().sum()

		print("[V] Epoch = %d : Loss = %f : Accuracy = %d/%d (%f)" % (epoch, loss/val_data.shape[0], correct, 
					val_data.shape[0], float(correct) * 100.0 / val_data.shape[0] ))

		print("")



def main():
	input_data, input_labels = get_data("../data/dataset/", device)
	input_data = normalize_data(input_data).float()
	train_data, train_labels, val_data, val_labels = split_data(input_data, input_labels)
	train(model, optim, train_data, train_labels, val_data, val_labels)


if __name__ == '__main__':
	main()