import torch
import Tanh

class RNN:

	def __init__(self):
		pass

	def forward(self, data, h, Wxh, Whh, B):
		return Tanh.forward(torch.matmul(data, Wxh)+torch.matmul(h, Whh)+B)

	def backward(self, Wxh, Whh, B, hid, out, grad_h, grad_o):
		# returns Wxh, Whh, B, grad_h, grad_o

		
