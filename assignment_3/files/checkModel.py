import sys
from optparse import OptionParser
import torch
import torchfile
sys.path.insert(0, '../src')
from Model import Model
from Linear import Linear
from ReLU import ReLU

USAGE_STRING = """"""
parser = OptionParser()

parser.add_option("--config",help="path to modelConfig.txt",default="modelConfig.txt")
parser.add_option("--input",help="path to input.bin",default="input.bin")
parser.add_option("--gradOutput",help="path to gradOutput.bin",default="gradOutput.bin")
parser.add_option("--output",help="path to output.bin",default="output.bin")
parser.add_option("--gradWeight",help="path to gradWeight.bin",default="gradWeight.bin")
parser.add_option("--gradB",help="path to gradB.bin",default="gradB.bin")
parser.add_option("--gradInput",help="path to gradInput.bin",default="gradInput.bin")

options, junk = parser.parse_args(sys.argv[1:])
if len(junk) != 0:
	raise Exception('Command line input not understood: ' + str(otherjunk))

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

lr = 0.001
model = Model(lr, "GradientDescent")

for i in range(2*num_layers-1):
	words = lines[i+1].split(' ')
	if words[0] == 'linear':
		model.addLayer(Linear(int(words[1]),int(words[2])))
	elif words[0] == 'relu':
		model.addLayer(ReLU())
model.set_device(device)

WL = torchfile.load(lines[2*num_layers][:-1])
BL = torchfile.load(lines[2*num_layers+1][:-1])

# print(len(model.Layers))
for i in range(num_layers):
	W = torch.DoubleTensor(WL[i])
	B = torch.DoubleTensor(BL[i])
	# print(model.Layers[i].W.shape)
	# print(model.Layers[i].B.shape)
	# print(W.shape)
	# print(B.shape)
	model.Layers[2*i].W = W
	model.Layers[2*i].B = B.reshape(B.shape[0],1)

out = torch.DoubleTensor(model.forward(inp))
# ref_out = torch.tensor(torchfile.load(options.output)).double()
# print(ref_out-out)
# torch.save(out,options.output)
