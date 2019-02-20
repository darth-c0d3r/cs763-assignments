import torch
<<<<<<< HEAD
import math
from Optimizer import *

class Linear:
	
	def __init__(self, input_channels, output_channels, kernal_shape, stride):
			
		self.optim = None

		self.input_channels = input_channels
		self.output_channels = output_channels
		self.kernal_shape = kernal_shape
		self.stride = stride

		self.W = torch.randn((output_channels, input_channels, kernal_shape, kernal_shape)).double() * 0.01 # k * j
		self.B = torch.randn((output_channels, 1)).double() * 0.01 + 0.01 # k * 1

		self.gradW = None
		self.gradB = None

		self.momW = None 
		self.momB = None 

		self.gradInput = None
		self.output = None

		self.device = None


	def forward(self, inp):

		out_row = int((inp.shape[2] - self.kernal_shape)/self.stride + 1)
		out_col = int((inp.shape[3] - self.kernal_shape)/self.stride + 1)

		self.output = np.zeros((inp.shape[0], self.output_channels, out_row, out_col)).to(self.device)
		for i in range(out_row):
			for j in range(out_col):
				data = inp[:,:, i*self.stride : i*self.stride+self.kernal_shape, j*self.stride : j*self.stride+self.kernal_shape]
				data = data.reshape(inp.shape[0],-1)
				kern = np.transpose(self.W.reshape(self.output_channels,-1))
				bias = np.repeat(self.B.reshape(1,self.B.shape[0]), inp.shape[0], axis=0)
				
				self.output[:,:,i,j] = (np.matmul(data,kern)+bias).reshape((inp.shape[0], self.output_channels))

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
		self.device = device
		self.W = self.W.to(device)
		self.B = self.B.to(device)

	def set_optim(self, optim):
		self.optim = optim

	def set_wd(self, wd):
		self.weight_decay = wd
