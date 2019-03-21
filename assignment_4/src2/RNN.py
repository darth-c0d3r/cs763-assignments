import torch
from Tanh import Tanh

class RNN:

	def __init__(self, input_dim, hidden_dim, output_dim):
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		self.Wxh = self.init_weights((input_dim, hidden_dim))
		self.Whh = self.init_weights((hidden_dim, hidden_dim))
		self.Why = self.init_weights((hidden_dim, output_dim))
		self.Bhh = self.init_weights((hidden_dim))
		self.Bhy = self.init_weights((output_dim))

		self.dWxh = torch.zeros(self.Wxh.shape)
		self.dWhh = torch.zeros(self.Whh.shape)
		self.dWhy = torch.zeros(self.Why.shape)
		self.dBhh = torch.zeros(self.Bhh.shape)
		self.dBhy = torch.zeros(self.Bhy.shape)

		self.h_t = None	# list of input from left cell (including output from rightmost cell) 
		self.y_t = None # list of output to above layer
		self.x_t = None # list of inputs to this layer
		self.grad_ht = None
		self.grad_xt = None

		self.timesteps = None


	def forward(self, data):
		self.timesteps = data.shape[1]
		self.x_t = data
		self.h_t = torch.zeros((data.shape[0], self.timesteps+1, self.hidden_dim)) # data.shape[0] = batch_size
		self.y_t = torch.zeros((data.shape[0], self.timesteps, self.output_dim))

		h = self.h_t[:,0,:]
		for time in range(self.timesteps):
			self.h_t[:,time+1,:] = Tanh.forward(torch.matmul(data[:,time,:], self.Wxh)+torch.matmul(h, self.Whh)+self.Bhh)
			h = self.h_t[:,time+1,:]
			self.y_t[:,time,:] = torch.matmul(h, self.Why)+self.Bhy
		
		return self.y_t


	def backward(self, grad_yt, lr):
		self.grad_ht = torch.zeros(self.h_t.shape)
		self.grad_xt = torch.zeros(self.x_t.shape)
		
		
		for time in range(self.timesteps-1,-1,-1):
			self.dWhy += torch.matmul(self.h_t[:,time+1,:].transpose(0,1), grad_yt[:,time,:])
			self.dBhy += torch.sum(grad_yt[:,self.timesteps-1,:], dim=0)

			dE_dht = torch.matmul(grad_yt[:,time,:], self.Why.transpose(0,1)) + self.grad_ht[:,time+1,:]
			act_back = torch.mul(dE_dht, Tanh.backward(self.h_t[:,time+1,:]))

			self.dWhh += torch.matmul(self.h_t[:,time,:].transpose(0,1), act_back)
			self.dWxh += torch.matmul(self.x_t[:,time,:].transpose(0,1), act_back)
			self.dBhh += torch.sum(act_back, dim=0)

			self.grad_xt[:,time,:] = torch.matmul(act_back, self.Wxh.transpose(0,1))
			self.grad_ht[:,time,:] = torch.matmul(act_back, self.Whh.transpose(0,1))
		
		self.Why -= lr * self.dWhy
		self.Bhy -= lr * self.dBhy
		self.Whh -= lr * self.dWhh
		self.Wxh -= lr * self.dWxh
		self.Bhh -= lr * self.dBhh

		return self.grad_xt

# TODO: Add clear_all to avoid exploding gradients

	def init_weights(self, dim):
		return torch.randn(dim)
