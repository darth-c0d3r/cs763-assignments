import sys
sys.path.append('./src')
import torch
import torchfile
import argparse
from Criterion import CrossEntropy

USAGE_STRING = """Arguments:\n(a) -i /path/to/input.bin\n(b) -t /path/to/target.bin\n(c) -ig /path/to/gradInput.bin""" 

parser = argparse.ArgumentParser(USAGE_STRING)

parser.add_argument("-i", "--input", help="Path to input.bin")
parser.add_argument("-t", "--target", help="Path to target.bin")
parser.add_argument("-ig", "--gradInput", help="Path to gradInput.bin")

args = parser.parse_args()

inp = torch.tensor(torchfile.load(args.input)).double()
target = torch.tensor(torchfile.load(args.target)).long()
target -= 1

loss = CrossEntropy.forward(inp, target)
print(loss)

gradInp = CrossEntropy.backward(inp, target)
torch.save(gradInp.double(), args.gradInput)
