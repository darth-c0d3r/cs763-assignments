import torch

class ReLU:

	def __init__(self):

		self.output = None
		self.gradInput = None
		
		self.name = "relu\n"

	def forward(self, inp):
		# if input = n x d
		self.output = inp # torch.max(inp, torch.zeros(inp.size()).double())
		self.output[self.output < 0] = 0
		return self.output

	def backward(self, inp, gradOutput, lr=None):
		dReLU = self.output
		dReLU[dReLU > 0] = 1
		self.gradInput = gradOutput * dReLU
		return self.gradInput

	def set_device(self, device):
		pass

	def set_optim(self, optim):
		pass

	def set_wd(self, wd):
		pass