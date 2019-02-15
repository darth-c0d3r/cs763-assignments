import torch
from Linear import Linear
from ReLU import ReLU
from Model import Model
from Criterion import CrossEntropy
import torchfile
import math

cuda = 1
device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu")
print("Device:", device)

data_folder = "../data/dataset/"

input_data = torchfile.load(data_folder+"train/data.bin")
input_labels = torchfile.load(data_folder+"train/labels.bin")

input_data = torch.tensor(input_data.reshape(input_data.shape[0],-1)).double().to(device)
input_labels = torch.tensor(input_labels.reshape(input_labels.shape[0],-1)).long().to(device)
input_data = input_data / torch.max(input_data)

train_data = input_data[:int(0.9 * input_data.shape[0]),:]
train_labels = input_labels[:int(0.9 * input_labels.shape[0]),:]
val_data = input_data[int(0.9 * input_data.shape[0]):,:]
val_labels = input_labels[int(0.9 * input_labels.shape[0]):,:]

print("Training Set Size: %d" % (train_data.shape[0]))
print("Validation Set Size: %d" % (val_data.shape[0]))
print("Input Size: %d" % (train_data.shape[1]))

batch_size = 500
epochs = 10

lr = 0.01
model = Model(lr)
model.addLayer(Linear(train_data.shape[1], 1024))
model.addLayer(ReLU())
model.addLayer(Linear(1024, 256))
model.addLayer(ReLU())
model.addLayer(Linear(256, 6))
model.set_device(device)

for epoch in range(epochs):

	for i in range(int(math.ceil(train_data.shape[0]/batch_size))):

		data = train_data[batch_size*(i) : batch_size*(i+1), :]
		target = train_labels[batch_size*(i) : batch_size*(i+1), :]

		out = model.forward(data)
		pred = torch.max(out, 1)[1]
		accuracy = torch.sum(pred.reshape(-1) == target.reshape(-1))

		loss = CrossEntropy.forward(out, target)
		model.backward(CrossEntropy.backward(out, target))
		model.setLearningRate(lr)
		print("Epoch = %d : Loss = %f : Accuracy = %d/%d" % (epoch, loss, accuracy, data.shape[0]))