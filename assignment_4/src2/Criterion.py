import torch

class OutputLayer:

	def __init__(self, input_dim, output_dim, lr):
		self.W = self.init_weight((input_dim, output_dim))
		self.B = self.init_weight((output_dim))
		self.lr = lr

	def forward(self, data, target):
		# data n x d
		# target n x 1
		self.inp = data
		self.out = torch.matmul(data, self.W) + self.B
		return self.out


	def backward(self, out_grad):
		in_grad = torch.matmul(out_grad, self.W.transpose(0,1))
		self.W -= lr * torch.matmul(self.inp.transpose(0,1), out_grad)
		# self.B -= lr * 

		return in_grad


class CrossEntropy:

	def forward(inp, target):
		# input - n x d
		# target 1D tensor n
		inp -= torch.mean(inp)
		exps = torch.exp(inp)
		sums = torch.sum(exps, 1)
		loss = torch.mean(-torch.log(torch.tensor([exps[i, target[i]]/sums[i] for i in range(len(target))])))
		return loss

	def backward(inp, target):
		inp -= torch.mean(inp)
		exps = torch.exp(inp)
		sums = torch.sum(exps, 1).reshape(-1,1).repeat(1, inp.size(1))
		grads = torch.div(exps, sums) 
		for i in range(len(target)):
			grads[i,target[i]] -= 1
		grads = grads/len(target)
		return grads.double()
