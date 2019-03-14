import torch
import math
from Optimizer import *

class Linear:
	
	def __init__(self, input_neurons, output_neurons):
			
		self.optim = None

		self.input_neurons = input_neurons
		self.output_neurons = output_neurons

		self.W = torch.randn((output_neurons, input_neurons)).double() / math.sqrt(input_neurons/2) # k * j
		self.B = torch.randn((output_neurons, 1)).double() / math.sqrt(input_neurons/2) + 0.01 # k * 1

		self.gradW = None # k * j
		self.gradB = None # k * 1

		self.momW = None # k * j
		self.momB = None # k * 1

		self.gradInput = None # n * j
		self.output = None # n * k

		self.name = "linear %d %d\n" % (input_neurons, output_neurons)


	def forward(self, inp):
		# input = n * j

		self.output = torch.matmul(inp, self.W.transpose(0,1)) + self.B.transpose(0,1)
		return self.output

	def backward(self, inp, gradOutput, lr):

		# input = n * j
		# gradOutput = n * k

		self.gradInput = torch.matmul(gradOutput, self.W)
		self.gradW = torch.matmul(gradOutput.transpose(0,1), inp)
		self.gradB = torch.sum(gradOutput, 0).reshape(-1, 1)

		if self.optim == "GradientDescent":
			GradientDescent(self, lr)
		elif self.optim == "GradientDescentWithMomentum":
			GradientDescentWithMomentum(self, lr)

		return self.gradInput

	def set_device(self, device):
		self.W = self.W.to(device)
		self.B = self.B.to(device)

	def set_optim(self, optim):
		self.optim = optim

	def set_wd(self, wd):
		self.weight_decay = wd