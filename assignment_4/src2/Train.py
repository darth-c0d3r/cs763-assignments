from utils import *
from Model import Model
from RNN import RNN
from Criterion import CrossEntropy

EPOCHS = 20
lr = 1e-1
BATCH_SIZE = 5

device = get_device(0)

train_dataset = Dataset("../data/train_data.txt",BATCH_SIZE,device)
train_labels = Dataset("../data/train_labels.txt",BATCH_SIZE,device,target=True)
test_dataset = Dataset("../data/test_data.txt",1,device)
train_dataset.print_details()

# print(train_dataset.get_batch(0).shape)

model = Model(1, 16, train_dataset.input_dim, 2, device)
criterion = CrossEntropy()

for epoch in range(1,EPOCHS+1):
	correct = 0
	loss = float(0)

	total = 0
	for batch_idx in range(train_dataset.num_batches):

		data = train_dataset.get_batch(batch_idx)
		target = train_labels.get_batch(batch_idx)

		total += data.shape[0]
	
		out_rnn_tot = model.forward(data)
		out_rnn = out_rnn_tot[:,-1,:]
		loss += criterion.forward(out_rnn, target).detach().item()
		
		pred = torch.argmax(out_rnn,1).reshape(-1,1)
		correct += torch.sum(torch.eq(pred, target)).detach().item()

		grad = torch.zeros(out_rnn_tot.shape).to(device)
		grad_out = criterion.backward(out_rnn, target)
		grad[:,-1,:] = grad_out
		
		model.backward(grad, lr)
		model.clear_grads()

	print("Epoch %d : Loss : %f : Correct : %d/%d (%f)" % (epoch, loss, correct, train_dataset.size, float(correct)/float(train_dataset.size)))
