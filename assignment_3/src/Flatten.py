import torch

class Flatten:

	def __init__(self):

		self.output = None
		self.gradInput = None
	
	def forward(self, inp):
		self.output = inp.reshape((inp.shape[0], -1)).double()
		return self.output

	def backward(self, inp, gradOutput, lr=None):
		self.gradInput = gradOutput.reshape(inp.shape).double()
		return self.gradInput

	def set_device(self, device):
		pass

	def set_optim(self, optim):
		pass

	def set_wd(self, wd):
		pass