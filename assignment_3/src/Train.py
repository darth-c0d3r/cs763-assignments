import torch
from Linear import Linear
from ReLU import ReLU
from Model import Model
from Criterion import CrossEntropy

import torchvision

def prepare_db():
	train_dataset = torchvision.datasets.MNIST('./data/mnist', train=True, download=True,
											   transform=torchvision.transforms.Compose([
												   torchvision.transforms.ToTensor(),
											   ]))

	eval_dataset = torchvision.datasets.MNIST('./data/mnist', train=False, download=True,
											   transform=torchvision.transforms.Compose([
												   torchvision.transforms.ToTensor(),
											   ]))
	return {'train':train_dataset,'eval':eval_dataset}

db = prepare_db()

# trainingData = torch.zeros((5000,10)).double()
# trainingData[:2500,:] = torch.randn((2500,10)) + 1
# trainingData[2500:,:] = torch.randn((2500,10)) - 1
# trainingLabels = torch.zeros((5000,1)).long()
# trainingLabels[:2500,:] = 1
# trainingData = trainingData.double()

batch_size = 6000
epochs = 10

lr = 0.1
model = Model(lr)
model.addLayer(Linear(784, 256))
# model.addLayer(ReLU())
model.addLayer(Linear(256, 10))

for epoch in range(epochs):
	train_loader = torch.utils.data.DataLoader(db['train'],batch_size=batch_size, shuffle=True)
	for batch_idx, (data, target) in enumerate(train_loader):
		data = data.reshape(data.shape[0],-1).double()
		
		out = model.forward(data)
		pred = torch.max(out, 1)[1]
		accuracy = torch.sum(pred == target)

		loss = CrossEntropy.forward(out, target)
		model.backward(CrossEntropy.backward(out, target))
		print("Epoch = %d : Loss = %f : Accuracy = %d/%d" % (epoch, loss, accuracy, data.shape[0]))
