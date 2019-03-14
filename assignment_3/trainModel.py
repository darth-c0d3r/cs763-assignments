import sys
sys.path.append("./src")

import os
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
import argparse

USAGE_STRING = """Arguments:\n(a) -i /path/to/input.bin\n(b) -t /path/to/target.bin\n(c) -ig /path/to/gradInput.bin""" 

parser = argparse.ArgumentParser(USAGE_STRING)

parser.add_argument("-modelName", "--modelName", help="Path to Model dir")
parser.add_argument("-data", "--dataPath", help="Path to train_data.bin")
parser.add_argument("-target", "--targetPath", help="Path to target_labels.bin")
args = parser.parse_args()

device = get_device(1)

input_data, input_labels = get_data(args.dataPath, args.targetPath, device, False)
input_data = normalize_data(input_data)
train_data, train_labels, val_data, val_labels = split_data(input_data, input_labels)

batch_size = 512
epochs = 100

# lr = [0.9,0.05] # lr, friction
lr = [0.001, 0.9]
model = Model(lr, "GradientDescentWithMomentum", 0.001)
model.addLayer(Linear(train_data.shape[1], 1024))
# model.addLayer(Dropout(0.4))
model.addLayer(ReLU())
model.addLayer(Linear(1024, 256))
# model.addLayer(Dropout(0.4))
model.addLayer(ReLU())
model.addLayer(Linear(256, 128))
model.addLayer(ReLU())
model.addLayer(Linear(128, 64))
model.addLayer(ReLU())
model.addLayer(Linear(64, 6))

model.set_device(device)

for epoch in range(epochs):

	loss_total = float(0)
	num_samples = 0
	num_correct = 0
	num_runs = 0

	if (epoch+1)%20 == 0:
		lr[0] /= 2.0

	for i in range(int(math.ceil(train_data.shape[0]/batch_size))):

		data = train_data[batch_size*(i) : batch_size*(i+1), :]
		target = train_labels[batch_size*(i) : batch_size*(i+1), :]

		out = model.forward(data)
		pred = torch.max(out, 1)[1]
		accuracy = torch.sum(pred.reshape(-1) == target.reshape(-1))

		loss = CrossEntropy.forward(out, target)
		gradOut = CrossEntropy.backward(out, target)
		model.backward(gradOut)
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

model.clearGradParam()

if args.modelName not in os.listdir():
	os.mkdir(args.modelName)
model.save(args.modelName)
