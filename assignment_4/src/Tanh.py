import torch

class Tanh:
	def __init__(self):
		self.output = None
		self.input = None
		self.gradInput = None
		self.name = "Tanh\n"

	def forward(self, inp):
		self.input = inp
		self.output = torch.tanh(inp)
		return self.output

	def backward(self, gradOutput):
		x = self.input
		self.gradInput =  gradOutput*(2*torch.exp(x)/(torch.exp(2*x)+1))**2
		return self.gradInput
