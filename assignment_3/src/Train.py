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

data_folder = "../data/dataset/train/"
input_data, input_labels = get_data(data_folder, device, False)
input_data = normalize_data(input_data)
train_data, train_labels, val_data, val_labels = split_data(input_data, input_labels)

batch_size = 512
epochs = 60

# lr = [0.9,0.05] # lr, friction
lr = [0.005, 0.9]
model = Model(lr, "GradientDescentWithMomentum", 0.001)
model.addLayer(Linear(train_data.shape[1], 1024))
model.addLayer(Dropout(0.4))
model.addLayer(ReLU())
model.addLayer(Linear(1024, 256))
model.addLayer(Dropout(0.4))
model.addLayer(ReLU())
model.addLayer(Linear(256, 6))
model.set_device(device)

for epoch in range(epochs):

	loss_total = float(0)
	num_samples = 0
	num_correct = 0
	num_runs = 0

	if (epoch + 1) % 15 == 0:
		lr[0] /= 4.0

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
		num_runs +=	1

	print("[T] Epoch = %d : Loss = %f : Accuracy = %d/%d (%f)" % (epoch, loss_total/num_runs, num_correct, 
				num_samples, float(num_correct) * 100.0 / float(num_samples) ))

	out = model.forward(val_data)
	pred = torch.max(out, 1)[1]
	accuracy = torch.sum(pred.reshape(-1) == val_labels.reshape(-1))
	loss = CrossEntropy.forward(out, val_labels)

	print("[V] Epoch = %d : Loss = %f : Accuracy = %d/%d (%.6f)" % (epoch, loss, accuracy, 
				val_data.shape[0], float(accuracy) * 100.0 / float(val_data.shape[0])))

	print("")


device = get_device(0)
model.set_device(device)
test_data = torchfile.load("../data/dataset/test/test.bin")
test_data = torch.tensor(test_data.reshape(test_data.shape[0],-1)).double().to(device)
test_data = normalize_data(test_data)
test_out  = model.forward(test_data)
test_pred = torch.max(test_out, 1)[1].reshape(-1)


with open("pred.csv", "w+") as file:
	file.write("id,label\n\n")
	for i in range(test_pred.shape[0]):
		file.write("%d,%d\n"%(i,test_pred[i]))
