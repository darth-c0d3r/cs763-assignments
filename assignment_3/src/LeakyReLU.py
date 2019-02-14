import torch

class LeakyReLU:

	def __init__(alpha):
		self.alpha = alpha
		self.output = None
		self.gradInput = None
	
	def forward(self, inp):
		# if input = n x d
		alph_inp = inp * self.alpha
		self.output = torch.max(inp, alph_inp)
		return self.output

	def backward(self, inp, gradOutput):
		dLReLU = inp
		dLReLU[> 0] = 1
		dLReLU[< 0] = self.alpha
		self.gradInput = gradOutput * dLReLU
		return self.gradInput
