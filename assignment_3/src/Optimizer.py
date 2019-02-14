import torch

def GradientDescent(layer, lr):
	layer.W -= lr * layer.gradW
	layer.B -= lr * layer.gradB

