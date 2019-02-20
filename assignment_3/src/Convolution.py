import torch
import math
from Optimizer import *

class Linear:
	
	def __init__(self, input_channels, output_channels, kernel_shape, stride):
			
		self.optim = None

		self.input_channels = input_channels
		self.output_channels = output_channels
		self.kernel_shape = kernel_shape
		self.stride = stride

		self.W = torch.randn((output_channels, input_channels, kernel_shape, kernel_shape)).double() * 0.01 # k * j
		self.B = torch.randn((output_channels, 1)).double() * 0.01 + 0.01 # k * 1

		self.gradW = None
		self.gradB = None

		self.momW = None 
		self.momB = None 

		self.gradInput = None
		self.output = None

		self.device = None


	def forward(self, inp):

		out_row = int((inp.shape[2] - self.kernel_shape)/self.stride + 1)
		out_col = int((inp.shape[3] - self.kernel_shape)/self.stride + 1)

		self.output = torch.zeros((inp.shape[0], self.output_channels, out_row, out_col)).to(self.device)
		for i in range(out_row):
			for j in range(out_col):
				data = inp[:,:, i*self.stride : i*self.stride+self.kernel_shape, j*self.stride : j*self.stride+self.kernel_shape]
				data = data.reshape(inp.shape[0],-1)
				kern = torch.transpose(self.W.reshape(self.output_channels,-1))
				bias = torch.repeat(self.B.reshape(1,self.B.shape[0]), inp.shape[0], axis=0)
				
				self.output[:,:,i,j] = (torch.matmul(data,kern)+bias).reshape((inp.shape[0], self.output_channels))

		return self.output

	def backward(self, inp, gradOutput, lr):

		# input = n * j
		# gradOutput = n * k

		out_row = int((inp.shape[2] - self.kernel_shape)/self.stride + 1)
		out_col = int((inp.shape[3] - self.kernel_shape)/self.stride + 1)

		gradInput = torch.zeros(inp.shape).to(self.device)
		self.gradW = torch.zeros(self.W.shape)
		self.gradB = torch.zeros(self.B.shape)
		for i in range(out_row):
			i_range = i*self.stride:i*self.stride+self.kernel_shape
			for j in range(out_col):
				j_range = j*self.stride,j*self.stride+self.kernel_shape
				for d in range(self.output_channels):
					gradInput[:,:,i_range,j_range] += gradOutput[:,d,i,j].reshape(inp.shape[0],1,1,1)*self.W[d].reshape(1,self.input_channels,self.kernel_shape,self.kernel_shape)
					self.gradW[d] += torch.sum(gradOutput[:,d,i,j].reshape(inp.shape[0],1,1,1)*inp[:,:,i_range,j_range],axis=0)

		self.gradB = torch.sum(temp,axis=(0,2,3))

		if self.optim == "GradientDescent":
			GradientDescent(self, lr)
		elif self.optim == "GradientDescentWithMomentum":
			GradientDescentWithMomentum(self, lr)

		return gradInput

	def set_device(self, device):
		self.device = device
		self.W = self.W.to(device)
		self.B = self.B.to(device)

	def set_optim(self, optim):
		self.optim = optim

	def set_wd(self, wd):
		self.weight_decay = wd
