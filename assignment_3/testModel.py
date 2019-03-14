import sys
sys.path.append('./src')
import torch
import torchfile
import argparse
from Model import Model
from util import *
from Linear import Linear
from ReLU import ReLU

parser = argparse.ArgumentParser()

parser.add_argument("-modelName", "--modelName", help="Path to best Model")
parser.add_argument("-data", "--data", help="Path to test.bin")

args = parser.parse_args()

config = open(args.modelName+"/modelConfig.txt", "r")
lines = config.readlines()
num_layers = int(lines[0])

model = Model([0.005, 0.9], "GradientDescentWithMomentum", 0.001)

for i in range(2*num_layers-1):
	words = lines[i+1].rstrip().split(' ')
	if words[0] == 'linear':
		model.addLayer(Linear(int(words[1]),int(words[2])))
	elif words[0] == 'relu':
		model.addLayer(ReLU())

WL = torch.load(args.modelName+"/"+lines[2*num_layers][:-1])
BL = torch.load(args.modelName+"/"+lines[2*num_layers+1][:-1])

cnt = 0
for i in range(len(model.Layers)):
	if not(hasattr(model.Layers[i],'W')):
		continue
	W = torch.DoubleTensor(WL[cnt])
	B = torch.DoubleTensor(BL[cnt])
	model.Layers[i].W = W
	model.Layers[i].B = B.reshape(B.shape[0],1)
	cnt = cnt+1

device = get_device(0)
model.set_device(device)
test_data = torchfile.load(args.data)
test_data = torch.tensor(test_data.reshape(test_data.shape[0],-1)).double().to(device)
test_data = normalize_data(test_data)
test_out  = model.forward(test_data)

test_pred = torch.max(test_out, 1)[1].reshape(-1)
torch.save(test_pred, "testPrediction.bin")

with open("pred.csv", "w+") as file:
	file.write("id,label\n\n")
	for i in range(test_pred.shape[0]):
		file.write("%d,%d\n"%(i,test_pred[i]))

