import torch

class Criterion(object):

	def forward(inp, target):
		# input - n x d
		# target 1D tensor n
		exps = torch.exp(inp)
		sums = torch.sum(exps, 1)
		loss = -torch.log(torch.tensor([exps[i, target[i]]/sums[i] for i in range(len(target))])) / len(target)
		return loss

	def backward(inp, target):
		exps = torch.exp(inp)
		sums = torch.sum(exps, 1).reshape(-1,1).repeat(inp.size(0), inp.size(1))
		grads = torch.div(exps, sums) 
		for i in range(len(target)):
			grads[i,target[i]] -= 1
		return grads
