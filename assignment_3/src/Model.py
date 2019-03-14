from Linear import Linear
from ReLU import ReLU
import torch

class Model:
	
	def __init__(self, lr, optim, weight_decay=0.0):
		self.Layers = list()
		self.isTrain = None
		self.optim = optim
		self.lr = lr
		self.weight_decay = weight_decay
		self.activations = list()

	def forward(self, inp):
		out = inp
		self.activations = [out]
		for layer in self.Layers:
			out = layer.forward(out)
			self.activations.append(out)
		return out

	def backward(self, gradOutput):
		gradInput = gradOutput
		for i in range(len(self.Layers)-1,-1,-1):
			gradInput = self.Layers[i].backward(self.activations[i], gradInput, self.lr)
		return gradInput

	def dispGradParam(self):
		for i in range(len(self.Layers)-1,-1,-1):
			if hasattr(layer, "W"):
				print(self.Layers[i].gradW)
				print(self.Layers[i].gradB)


	def clearGradParam(self):
		for layer in self.Layers:
			if hasattr(layer, "W"):
				layer.gradW = None
				layer.gradB = None
				layer.momW = None
				layer.momB = None
			layer.gradInput = None
			layer.output = None

	def save(self, modelName):
		weights = list()
		biases = list()
		with open(modelName+"/modelConfig.txt", "w+") as file:
			file.write("%d\n" % ((len(self.Layers)+1)//2))
			for layer in self.Layers:
				if hasattr(layer, "W"):
					weights.append(layer.W.cpu())
					biases.append(layer.B.cpu())
				file.write(layer.name)
			file.write("weights.bin\n")
			file.write("biases.bin\n")	
		torch.save(weights, modelName+"/weights.bin")
		torch.save(biases, modelName+"/biases.bin")

	def addLayer(self, layer):
		self.Layers.append(layer)
		self.Layers[-1].set_optim(self.optim)
		self.Layers[-1].set_wd(self.weight_decay)

	def setLearningRate(self, lr):
		self.lr = lr

	def set_device(self, device):
		for layer in self.Layers:
			layer.set_device(device)
