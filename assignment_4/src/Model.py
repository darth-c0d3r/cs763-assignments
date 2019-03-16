import torch

class Model:

	def __init__(self, num_layers, hidden_dim, input_dim, output_dim):

		self.num_layers = num_layers
		self.hidden_dim = hidden_dim
		self.input_dim = input_dim
		self.output_dim = output_dim

		self.outputs = None

		self.Wxh = []
		self.Whh = []
		self.B = []

		fc = [input_dim] + [hidden_dim]*num_layers

		for i in range(self.num_layers):
			self.Wxh.append(self.init_weights((fc[i], fc[i+1])))
			self.Whh.append(self.init_weights((hidden_dim, hidden_dim)))
			self.B.append(self.init_weights((hidden_dim)))

		self.Why = self.init_weights((hidden_dim, output_dim))
		self.Bhy = self.init_weights((output_dim))


	def forward(self, data, hidden_vectors):

		self.outputs = torch.zeros((data.shape[0], data.shape[1], self.num_layers, self.hidden_dim))

		for time in range(data.shape[1]):
			data_ = data[:,time,:].reshape(data.shape[0], data.shape[2])

			for idx in range(self.num_layers): 

				hidden_vectors[:,idx] = torch.matmul(data_, self.Wxh[idx]) + torch.matmul(hidden_vectors[:,idx],self.Whh[idx]) + self.B[idx] # add activation
				data_ = hidden_vectors[:,idx]

			self.outputs[:,time,:,:] = hidden_vectors

		data_ = self.outputs[:,-1,-1,:].reshape(data.shape[0], self.hidden_dim)
		return torch.matmul(data_,self.Why) + self.Bhy

	def init_weights(self, dim):
		return torch.randn(dim)