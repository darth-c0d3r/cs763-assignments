import torch

class CrossEntropy:

	def __init__(self):
		pass

	def forward(self,inp,target):
		# input - n x d
		# target 1D tensor n
		inp -= torch.mean(inp)
		exps = torch.exp(inp)
		sums = torch.sum(exps, 1)
		loss = torch.mean(-torch.log(torch.tensor([exps[i, target[i]]/sums[i] for i in range(len(target))])))
		return loss

	def backward(self,inp, target):
		inp -= torch.mean(inp)
		exps = torch.exp(inp)
		sums = torch.sum(exps, 1).reshape(-1,1).repeat(1, inp.size(1))
		grads = torch.div(exps, sums) 
		for i in range(len(target)):
			grads[i,target[i]] -= 1
		grads = grads/len(target)
		return grads.double()
