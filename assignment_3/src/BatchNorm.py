import torch
from Optimizer import *

class BatchNorm1D:

	def __init__(self, input_neurons, isTrain=True):

		self.optim = None

		self.input_neurons = input_neurons

		self.mean = None
		self.var = None
		self.currMean = None
		self.currVar = None
		self.normalizedInput = None

		self.decay = 0.9
		self.isTrain = isTrain

		self.W = torch.ones((1,input_neurons)).double()
		self.B = torch.zeros((1,input_neurons)).double()

		self.gradW = None
		self.gradB = None

		self.momW = torch.zeros((1,input_neurons)).double()
		self.momB = torch.zeros((1,input_neurons)).double()

		self.gradInput = None # n * j
		self.output = None # n * k

	def forward(self, inp):

		if self.isTrain:
			mean = torch.mean(inp, 0).reshape(1,-1) # 1 * k
			var = torch.sum(((inp-mean)**2),0).reshape(1,-1) / inp.shape[0]
			self.currMean, self.currVar = mean, var

			if self.mean is None:
				self.mean = mean
				self.var = var
			else:
				self.mean = self.decay * self.mean + (1.0 - self.decay) * mean
				self.var  = self.decay * self.var  + (1.0 - self.decay) * var

			self.normalizedInput = (inp - mean) / torch.sqrt(var + 1e-7)
			self.output = self.W * self.normalizedInput + self.B

		else:
			self.normalizedInput = (inp - self.mean) / torch.sqrt(self.var + 1e-7)
			self.output = self.W * self.normalizedInput + self.B

		return self.output

	def backward(self, inp, gradOutput, lr):

		self.gradW = gradOutput * self.normalizedInput
		self.gradB = gradOutput

		self.gradInput = self.W / torch.sqrt(self.currVar + 1e-7)
		self.gradInput = self.gradInput.repeat(inp.shape[0],1)

		self.gradW = torch.sum(self.gradW, 0).reshape(1,-1)
		self.gradB = torch.sum(self.gradB, 0).reshape(1,-1)

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