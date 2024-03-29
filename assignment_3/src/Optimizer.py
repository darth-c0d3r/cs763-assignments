import torch

def GradientDescent(layer, lr):
	# len(lr) = 1
	layer.W -= lr[0] * ( layer.gradW + 2*layer.weight_decay*layer.W)
	layer.B -= lr[0] * layer.gradB 

def GradientDescentWithMomentum(layer, lr):
	# len(lr) = 2
	if layer.momW is None:
		layer.momW = lr[0] * ( layer.gradW + 2*layer.weight_decay*layer.W )
		layer.momB = lr[0] * layer.gradB
	else:
		layer.momW = lr[1]*layer.momW + lr[0]*(layer.gradW + 2*layer.weight_decay*layer.W )
		layer.momB = lr[1]*layer.momB + lr[0]*layer.gradB
	layer.W -= layer.momW
	layer.B -= layer.momB