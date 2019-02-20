import sys
import argparse
import torch
import torchfile
sys.path.insert(0, '../src')
from Model import Model
from Linear import Linear
from ReLU import ReLU

USAGE_STRING = """Arguments:\n(a) -i /path/to/input.bin\n(b) -t /path/to/target.bin\n(c) -ig /path/to/gradInput.bin""" 
parser = argparse.ArgumentParser(USAGE_STRING)


parser.add_argument("-config","--config",help="path to modelConfig.txt",default="modelConfig.txt")
parser.add_argument("-i","--input",help="path to input.bin",default="input.bin")
parser.add_argument("-og","--gradOutput",help="path to gradOutput.bin",default="gradOutput.bin")
parser.add_argument("-o","--output",help="path to output.bin",default="output.bin")
parser.add_argument("-ow","--gradWeight",help="path to gradWeight.bin",default="gradWeight.bin")
parser.add_argument("-ob","--gradB",help="path to gradB.bin",default="gradB.bin")
parser.add_argument("-ig","--gradInput",help="path to gradInput.bin",default="gradInput.bin")
options = parser.parse_args()

cuda = 1
device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu")

inp = torch.tensor(torchfile.load(options.input)).double().to(device)
gradOutput = torch.tensor(torchfile.load(options.gradOutput))

config = open(options.config, "r")
lines = config.readlines()
num_layers = int(lines[0])
batch_size = inp.shape[0]
inp = inp.reshape(inp.shape[0],-1)
# inp = inp/torch.max(inp)

lr = [0.001]
model = Model(lr, "GradientDescent")

for i in range(2*num_layers-1):
	words = lines[i+1].rstrip().split(' ')
	if words[0] == 'linear':
		model.addLayer(Linear(int(words[1]),int(words[2])))
	elif words[0] == 'relu':
		model.addLayer(ReLU())
model.set_device(device)

WL = torchfile.load(lines[2*num_layers][:-1])
BL = torchfile.load(lines[2*num_layers+1][:-1])

cnt = 0
for i in range(len(model.Layers)):
	if not(hasattr(model.Layers[i],'W')):
		continue
	W = torch.DoubleTensor(WL[cnt])
	B = torch.DoubleTensor(BL[cnt])
	model.Layers[i].W = W
	model.Layers[i].B = B.reshape(B.shape[0],1)
	cnt = cnt+1

out = torch.DoubleTensor(model.forward(inp))
torch.save(out,options.output)
gradInput = model.backward(gradOutput)
torch.save(gradInput,options.gradInput)

gradW = list()
gradB = list()
for i in range(len(model.Layers)):
	if not(hasattr(model.Layers[i],'gradW')):
		continue
	gradW.append(model.Layers[i].gradW)
	gradB.append(model.Layers[i].gradB.reshape(-1))

torch.save(gradW,options.gradWeight)
torch.save(gradB,options.gradB)
