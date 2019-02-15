import torch

def GradientDescent(layer, lr):
	layer.W -= lr[0] * layer.gradW
	layer.B -= lr[0] * layer.gradB

def GradientDescentWithMomentum(layer, lr):
	layer.momW = lr[0]*layer.momW + lr[1]*layer.gradW
	layer.momB = lr[0]*layer.momB + lr[1]*layer.gradB
	layer.W -= layer.momW
	layer.B -= layer.momB

