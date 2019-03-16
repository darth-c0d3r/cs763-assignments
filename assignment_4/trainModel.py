import os
import sys
import argparse
import torch
import torchfile
sys.path.append('./src')
from train import *
from utils import *
from Tanh import *

parser = argparse.ArgumentParser()


parser.add_argument("-modelName" "--model", help="name of model to be saved")
parser.add_argument("-data","--data",help="path to train_data.txt")
parser.add_argument("-target","--target",help="path to train_labels.txt")

options = parser.parse_args()

device = get_device()

model = train(options.data, options.target, device)

os.system('mkdir '+options.model)
torch.save(model, options.model+'/'+options.model)
