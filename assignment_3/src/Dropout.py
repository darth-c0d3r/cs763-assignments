import torch

class Dropout:
	def __init__(self, dropout_rate, isTrain=True):
		self.dropout_rate = dropout_rate
		self.not_dropped = None
		self.output = None
		self.gradInput = None
		self.isTrain = isTrain
		self.device = None

	def forward(self, inp):
		self.output = inp
		if (self.isTrain):
			rand_vals = torch.rand(inp.shape).double() # values in [0,1]
			self.not_dropped = ((rand_vals > self.dropout_rate).double()).to(self.device)
			self.output *= self.not_dropped/(1.0-self.dropout_rate)
		return self.output

	def backward(self, inp, gradOutput, lr=None):
		self.gradInput = gradOutput
		if (self.isTrain):
			self.gradInput *= self.not_dropped/(1.0-self.dropout_rate)
		return self.gradInput

	def set_device(self, device):
		self.device = device

	def set_optim(self, optim):
		pass

	def set_wd(self, wd):
		pass