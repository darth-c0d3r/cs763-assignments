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


	def forward(self, data):

		total_time = data.shape[1]
		batch_size = data.shape[0]

		self.outputs = torch.zeros((batch_size, total_time+1, self.num_layers+1, self.hidden_dim))
		self.outputs[:,:,0,:] = data

		hidden_vectors = torch.zeros((batch_size, self.num_layers, self.hidden_dim))

		for time in range(total_time):
			data_ = data[:,time,:].reshape(batch_size, self.input_dim)
			for idx in range(self.num_layers): 

				hidden_vectors[:,idx] = RNN.forward(data_, hidden_vectors[:,idx], self.Wxh[idx], self.Whh[idx], self.B[idx])
				data_ = hidden_vectors[:,idx]

			self.outputs[:,time+1,1:,:] = hidden_vectors

		data_ = self.outputs[:,-1,-1,:].reshape(batch_size, self.hidden_dim)
		return torch.matmul(data_,self.Why) + self.Bhy

	def backward(self, grad_out):

		total_time = self.outputs.shape[1]
		batch_size = self.outputs.shape[0]

		grad_hid = torch.zeros((self.num_layers, batch_size, self.hidden_dim))

		for time in range(total_time-1,0,-1):
			for idx in range(self.num_layers,0,-1):
				self.Wxh, self.Whh, self.B, grad_hid[idx-1], grad_out = RNN.backward(self.Wxh, self.Whh, self.B, self.outputs[:,time-1:idx,:], self.outputs[:,time,idx-1,:], grad_hid[idx-1], grad_out)
			grad_out = torch.zeros(grad_out.shape)

	def init_weights(self, dim):
		return torch.randn(dim)