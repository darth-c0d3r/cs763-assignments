import sys
sys.path.append('./src')
import torch
import torchfile
from optparse import OptionParser
from Criterion import CrossEntropy

USAGE_STRING = """Arguments:\n(a) -i /path/to/input.bin\n(b) -t /path/to/target.bin\n(c) -ig /path/to/gradInput.bin"""

parser = OptionParser(USAGE_STRING)

parser.add_option('-i', '--input', help='Path to input.bin', type="str")
parser.add_option('-t', '--target', help='Path to target.bin', type="str")
parser.add_option('-g', '--gradInput', help='Path to gradInput.bin', type="str") # DO -ig LATER BUT HOW 

options, otherjunk = parser.parse_args(sys.argv[1:])
if len(otherjunk) != 0: 
	print(USAGE_STRING)
	raise Exception('Command line input not understood: ' + str(otherjunk))

inp = torch.tensor(torchfile.load(options.input)).double()
target = torch.tensor(torchfile.load(options.target)).long()
target -= 1

loss = CrossEntropy.forward(inp, target)
print(loss)

gradInp = CrossEntropy.backward(inp, target)
torch.save(gradInp, options.gradInput)