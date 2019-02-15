import torch
from Optimizer import *

class Linear:
	
	def __init__(self, input_neurons, output_neurons):
			
		self.optim = None

		self.input_neurons = input_neurons
		self.output_neurons = output_neurons

		self.W = torch.randn((output_neurons, input_neurons)).double() * 0.1 # k * j
		self.B = torch.randn((output_neurons, 1)).double() * 0.1 # k * 1

		self.gradW = None # k * j
		self.gradB = None # k * 1

		self.momW = torch.zeros((output_neurons, input_neurons)).double()
		self.momB = torch.zeros((output_neurons, 1)).double()

		self.gradInput = None # n * j
		self.output = None # n * k


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
		self.momW = self.momW.to(device)
		self.momB = self.momB.to(device)

	def set_optim(self, optim):
		self.optim = optim