import os
import sys
import argparse
import torch
import torchfile
sys.path.append('./src')

from utils import *
from Model import Model
from Criterion import CrossEntropy

parser = argparse.ArgumentParser()

parser.add_argument("-modelName", "--modelName", help="name of model to be saved")
parser.add_argument("-data","--data",help="path to train_data.txt")
parser.add_argument("-target","--target",help="path to train_labels.txt")

options = parser.parse_args()

device = get_device()
EPOCHS = 30
lr = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4]
BATCH_SIZE = 2

train_dataset = Dataset(options.data, BATCH_SIZE, device)
train_labels = Dataset(options.target, BATCH_SIZE, device, target=True)
train_dataset.print_details()

def train():
	val_data = train_dataset.get_eval()
	val_target = train_labels.get_eval()

	model = Model(1, 16, train_dataset.input_dim, 2, device)
	criterion = CrossEntropy()

	for epoch in range(1,EPOCHS+1):
		correct = 0
		loss = float(0)

		total = 0
		for batch_idx in range(train_dataset.num_batches):

			data = train_dataset.get_batch(batch_idx)
			target = train_labels.get_batch(batch_idx)

			total += data.shape[0]
		
			out_rnn_tot = model.forward(data)
			out_rnn = out_rnn_tot[:,-1,:]
			loss += criterion.forward(out_rnn, target).detach().item()
			
			pred = torch.argmax(out_rnn,1).reshape(-1,1)
			correct += torch.sum(torch.eq(pred, target)).detach().item()

			grad = torch.zeros(out_rnn_tot.shape).to(device)
			grad_out = criterion.backward(out_rnn, target)
			grad[:,-1,:] = grad_out
			
			model.backward(grad, lr[(epoch-1)//6])
			model.clear_grads()

		print("Train %d : Loss : %f : Accuracy : %d/%d (%f)" %
			(epoch, loss/train_dataset.train_size, correct, train_dataset.train_size, float(correct)/float(train_dataset.train_size))
		)

		val_out = model.forward(val_data)[:,-1,:]
		val_pred = torch.argmax(val_out,1).reshape(-1,1)
		val_correct = torch.sum(torch.eq(val_pred, val_target)).detach().item()
		val_loss = criterion.forward(val_out, val_target).detach().item()

		print("Eval  %d : Loss : %f : Accuracy : %d/%d (%f)" % 
			(epoch, val_loss/train_dataset.val_size, val_correct, train_dataset.val_size, float(val_correct)/train_dataset.val_size)
		)

	model.make_empty()
	return model

model = train()

torch.save(model, 'bestModel/'+options.modelName)
