import sys
import argparse
import torch
import torchfile
sys.path.append('./src')

from utils import *
from Model import Model

parser = argparse.ArgumentParser()

parser.add_argument("-modelName", "--modelName", help="name of model to be loaded")
parser.add_argument("-data","--data",help="path to test_data.txt")

options = parser.parse_args()

device = get_device()

def test(model_t):
	test_data = one_hot_encode(get_data(test_dataset.data), get_unique())
	test_out = model_t.forward(test_data)[:,-1,:]
	test_pred = torch.argmax(test_out,1).reshape(-1,1)

	torch.save(test_pred, "testPrediction.bin")

	with open("pred.csv", "w+") as file:
		file.write("id,label\n\n")
		for i in range(test_pred.shape[0]):
			file.write("%d,%d\n"%(i,test_pred[i]))

test_dataset = Dataset(options.data, 1, device)
model = torch.load(options.modelName)

test(model)
