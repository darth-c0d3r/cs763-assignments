from Linear import Linear
from ReLU import ReLU

class Model:
	
	def __init__(self):
		
		self.Layers = list()
		self.isTrain = None
		self.lr = 0.1

	def forward(self, inp):
		out = inp
		for layer in self.Layers:
			out = layer.forward(out)
		return out


	def backward(self, activations, gradOutput):
		gradInput = gradOutput
		for i in range(len(self.Layers)-1,-1,-1):
			gradInput = layer.backward(activations[i], gradInput, self.lr)

	def dispGradParam(self):
		pass

	def clearGradParam(self):
		pass

	def addLayer(self, layer):
		self.Layers.append(layer)