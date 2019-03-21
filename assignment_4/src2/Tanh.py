import torch

class Tanh:
	def forward(inp):
		return torch.tanh(inp)

	def backward(inp):
		return 1 - inp**2
