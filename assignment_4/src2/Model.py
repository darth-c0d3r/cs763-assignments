import torch
from RNN import RNN

class Model:

	def __init__(self, num_layers, hidden_dim, input_dim, output_dim, device):

		self.device = device
		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.layers = []
		fc = [input_dim] + [hidden_dim]*num_layers + [output_dim]
		for i in range(self.num_layers):
			self.layers.append(RNN(fc[i], fc[i+1], fc[i+2],self.device))		

		self.outputs = None
		self.timesteps = None


	def forward(self, data):
		data_ = data
		self.timesteps = data.shape[1]
		for i in range(self.num_layers):
			data_ = self.layers[i].forward(data_)

		return data_

	def backward(self, grad_out, lr):
		grad_ = grad_out
		for i in range(self.num_layers-1,-1,-1):
			grad_ = self.layers[i].backward(grad_, lr)

	def clear_grads(self):
		for layer in self.layers:
			layer.clear_grads()