import sys
sys.path.append('./src')
import torch
import torchfile
import argparse
from Model import Model
from util import *

USAGE_STRING = """Arguments:\n(a) -i /path/to/input.bin\n(b) -t /path/to/target.bin\n(c) -ig /path/to/gradInput.bin""" 

parser = argparse.ArgumentParser(USAGE_STRING)

parser.add_argument("-modelName", "--modelName", help="Path to best Model")
parser.add_argument("-data", "--data", help="Path to test.bin")

args = parser.parse_args()


model = torch.load(args.modelName)

device = get_device(0)
model.set_device(device)
test_data = torchfile.load(args.data)
test_data = torch.tensor(test_data.reshape(test_data.shape[0],-1)).double().to(device)
test_data = normalize_data(test_data)
test_out  = model.forward(test_data)

test_pred = torch.max(test_out, 1)[1].reshape(-1)
torch.save(test_pred, "testPrediction.bin")
