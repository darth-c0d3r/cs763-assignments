import torch

class LeakyReLU:

	def __init__(self, alpha):
		self.alpha = alpha
		self.output = None
		self.gradInput = None
	
	def forward(self, inp):
		# if input = n x d
		alph_inp = inp * self.alpha
		self.output = torch.max(inp, alph_inp)
		return self.output

	def backward(self, inp, gradOutput, lr=None):
		dLReLU = inp
		dLReLU[dLReLU > 0] = 1
		dLReLU[dLReLU < 0] = self.alpha
		self.gradInput = gradOutput * dLReLU
		return self.gradInput

	def set_device(self, device):
		pass

	def set_optim(self, optim):
		pass