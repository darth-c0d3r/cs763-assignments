import torch
import Tanh

class RNN:

	def __init__(self):
		pass

	def forward(self, data, h, Wxh, Whh, B):
		return Tanh.forward(torch.matmul(data, Wxh)+torch.matmul(h, Whh)+B)

	def backward(self, Wxh, Whh, B, hid, out, grad_h, grad_o):
		# returns Wxh, Whh, B, grad_h, grad_o

		self.dWxh = torch.zeros(self.Wxh.shape)
		self.dWhh = torch.zeros(self.Whh.shape)
		self.dWhy = torch.zeros(self.Why.shape)
		self.dBhh = torch.zeros(self.Bhh.shape)
		self.dBhy = torch.zeros(self.Bhy.shape)

		self.grad_xt = torch.zeros((self.timesteps,input_dim))
		self.grad_ht = torch.matmul(grad_yt,self.Why.transpose(0,1))
		for time in range(self.timesteps-1,-1,-1):
			scaling = torch.diag(Tanh.backward(self.z[:,time,:]))
			# grad_ht[time,:] = torch.matmul(grad_yt[time,:],self.Why.transpose(0,1))
			grad_xt[time,:] = torch.matmul(grad_ht[time,:],torch.dot(scaling,self.Wxh.transpose(0,1)))

			#dwhy+= h_t.repeat(1, o) -> hxo multiply each row (1xo) by grad_y (1xo)
			self.dWhy += torch.matmul(grad_ht[time,:].transpose(0,1),grad_yt[time,:])
			self.dWxh += torch.matmul(grad_ht[time,:],torch.dot(scaling,grad_xt[time,:]))
			self.dBhh += torch.matmul(grad_ht[time,:],scaling)
			self.dBhy += grad_yt[time,:]

			grad_ht[time,:] = torch.matmul(grad_ht[time,:],torch.dot(scaling,self.Whh.transpose(0,1)))
			self.dWhh += torch.matmul(grad_ht[time,:],torch.dot(scaling,grad_ht[time,:]))

		return grad_xt
