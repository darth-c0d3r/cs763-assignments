import sys
import argparse
import torch
import torchfile
sys.path.append('./src')

parser = argparse.ArgumentParser()

parser.add_argument("-modelName" "--model", help="name of model to be loaded")
parser.add_argument("-data","--data",help="path to test_data.txt")

