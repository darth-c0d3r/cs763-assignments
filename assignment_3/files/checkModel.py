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

inp = inp / torch.max(inp)

config = open(options.config, "r")
lines = config.readlines()
num_layers = int(lines[0])
# print(inp.shape)
batch_size = inp.shape[0]


# epochs = 10
lr = 0.01
model = Model(lr)

for i in range(num_layers):
	words = lines[i+1].split(' ')
	if words[0] == 'linear':
		model.addLayer(Linear(int(words[1]),int(words[2])))
	elif words[0] == 'relu':
		model.addLayer(ReLU())
model.set_device(device)
# model.forward(inp)

# torch.save('')