from Linear import Linear
from ReLU import ReLU

class Model:
	
	def __init__(self, lr, optim):
		
		self.Layers = list()
		self.isTrain = None
		self.optim = optim
		self.lr = lr
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

	def dispGradParam(self):
		pass

	def clearGradParam(self):
		pass	

	def addLayer(self, layer):
		self.Layers.append(layer)
		self.Layers[-1].set_optim(self.optim)

	def setLearningRate(self, lr):
		self.lr = lr

	def set_device(self, device):
		for layer in self.Layers:
			layer.set_device(device)