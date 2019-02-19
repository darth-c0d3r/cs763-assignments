import torch
from Linear import Linear
from ReLU import ReLU
from Dropout import Dropout
from BatchNorm import BatchNorm1D
from Model import Model
from Criterion import CrossEntropy
import torchfile
import math
from util import *

device = get_device(1)

data_folder = "../data/dataset/"
input_data, input_labels = get_data(data_folder, device)
input_data = normalize_data(input_data)
train_data, train_labels, val_data, val_labels = split_data(input_data, input_labels)

batch_size = 500
epochs = 50

# lr = [0.9,0.05] # lr, friction
lr = [0.9, 0.01]
model = Model(lr, "GradientDescentWithMomentum")
model.addLayer(Linear(train_data.shape[1], 1024))
# model.addLayer(BatchNorm1D(1024))
model.addLayer(ReLU())
model.addLayer(Linear(1024, 6))
# model.addLayer(BatchNorm1D(512))
# model.addLayer(ReLU())
# model.addLayer(Linear(512, 256))
# model.addLayer(BatchNorm1D(256))
# model.addLayer(ReLU())
# model.addLayer(Linear(256, 6))
model.set_device(device)

for epoch in range(epochs):

	loss_total = float(0)
	num_samples = 0
	num_correct = 0

	for i in range(int(math.ceil(train_data.shape[0]/batch_size))):

		data = train_data[batch_size*(i) : batch_size*(i+1), :]
		target = train_labels[batch_size*(i) : batch_size*(i+1), :]

		out = model.forward(data)
		pred = torch.max(out, 1)[1]
		accuracy = torch.sum(pred.reshape(-1) == target.reshape(-1))

		loss = CrossEntropy.forward(out, target)
		model.backward(CrossEntropy.backward(out, target))
		model.setLearningRate(lr)

		loss_total += loss
		num_samples += data.shape[0]
		num_correct += accuracy

	print("[T] Epoch = %d : Loss = %f : Accuracy = %d/%d (%f)" % (epoch, loss_total/num_samples, num_correct, 
				num_samples, num_correct * 100.0 / num_samples ))

	out = model.forward(val_data)
	pred = torch.max(out, 1)[1]
	accuracy = torch.sum(pred.reshape(-1) == val_labels.reshape(-1))
	loss = CrossEntropy.forward(out, val_labels)

	print("[V] Epoch = %d : Loss = %f : Accuracy = %d/%d (%f)" % (epoch, loss, accuracy, 
				val_data.shape[0], accuracy * 100.0 / val_data.shape[0] ))

	print("")
