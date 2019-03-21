import torch
from RNN import RNN

class Model:

	def __init__(self, num_layers, hidden_dim, input_dim, output_dim):

		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.layers = []
		fc = [input_dim] + [hidden_dim]*num_layers + [output_dim]
		for i in range(self.num_layers):
			self.layers.append(RNN(fc[i], fc[i+1], fc[i+2]))		

		self.outputs = None


	def forward(self, data):
		data_ = data
		for i in range(self.num_layers):
			data_ = self.layers[i].forward(data_)

		return data_


	def backward(self, grad_out):
		grad_ = grad_out
		lr = 0.01
		for i in range(self.num_layers-1,-1,-1):
			grad_ = self.layers[i].backward(grad_, lr)
		
		return grad_
