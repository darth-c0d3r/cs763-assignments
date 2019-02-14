from Linear import Linear
from ReLU import ReLU

class Model:
	
	def __init__(self):
		
		self.Layers = list()
		self.isTrain = None

	def forward(self, inp):
		out = inp
		for layer in self.Layers:
			out = layer.forward(out)
		return out


	def backward(self, inp, gradOutput):
		pass

	def dispGradParam(self):
		pass

	def clearGradParam(self):
		pass

	def addLayer(self, layer):
		self.Layers.append(layer)