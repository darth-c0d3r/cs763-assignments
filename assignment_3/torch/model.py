import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np

class fc_net(nn.Module):

	def __init__(self, size, f_c, output_size):
		super(fc_net, self).__init__()
		
		self.input_size = size

		# fully connected layers
		f_c = [self.input_size] + f_c
		self.fully_connected = nn.ModuleList()
		self.dropout = nn.ModuleList()
		for i in range(len(f_c)-1):
			self.fully_connected.append(nn.Linear(f_c[i], f_c[i+1]))
			self.dropout.append(nn.Dropout(0.6))

		self.output_layer = nn.Linear(f_c[-1], output_size)

	def forward(self, x):

		# fully connected layers
		x = x.view(-1, self.input_size)
		for i in range(len(self.fully_connected)):
			x = torch.relu(self.dropout[i](self.fully_connected[i](x)))

		# output layer
		x = self.output_layer(x)

		return x