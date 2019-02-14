import torch
from Optimizer import *

class Linear:
	
	def __init__(self, input_neurons, output_neurons):
		
		self.input_neurons = input_neurons
		self.output_neurons = output_neurons

		self.W = torch.randn((output_neurons, input_neurons)).double() * 0.1 # k * j
		self.B = torch.randn((output_neurons, 1)).double() * 0.1 # k * 1

		self.gradW = torch.zeros((output_neurons, input_neurons)).double() # k * j
		self.gradB = torch.zeros((output_neurons, 1)).double() # k * 1

		self.gradInput = None # n * j
		self.output = None # n * k


	def forward(self, inp):
		# input = n * j

		self.output = torch.matmul(inp, self.W.transpose(0,1)) + self.gradB.transpose(0,1)
		return self.output

	def backward(self, inp, gradOutput, lr):

		# input = n * j
		# gradOutput = n * k

		self.gradInput = torch.matmul(gradOutput, self.W)
		self.gradW = torch.matmul(gradOutput.transpose(0,1), inp)
		self.gradB = torch.sum(gradOutput, 0).reshape(-1, 1)

		GradientDescent(self, lr)

		return self.gradInput
