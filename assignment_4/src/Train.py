from utils import *
from Model import Model
from RNN import RNN
from Criterion import CrossEntropy

EPOCHS = 30
lr = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4]
BATCH_SIZE = 2

device = get_device(0)

train_dataset = Dataset("../data/train_data.txt",BATCH_SIZE,device)
train_labels = Dataset("../data/train_labels.txt",BATCH_SIZE,device,target=True)
test_dataset = Dataset("../data/test_data.txt",1,device)
train_dataset.print_details()

def train():
	val_data = train_dataset.get_eval()
	val_target = train_labels.get_eval()

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
			
			model.backward(grad, lr[(epoch-1)//6])
			model.clear_grads()

		print("Train %d : Loss : %f : Accuracy : %d/%d (%f)" %
			(epoch, loss/train_dataset.train_size, correct, train_dataset.train_size, float(correct)/float(train_dataset.train_size))
		)

		val_out = model.forward(val_data)[:,-1,:]
		val_pred = torch.argmax(val_out,1).reshape(-1,1)
		val_correct = torch.sum(torch.eq(val_pred, val_target)).detach().item()
		val_loss = criterion.forward(val_out, val_target).detach().item()

		print("Eval  %d : Loss : %f : Accuracy : %d/%d (%f)" % 
			(epoch, val_loss/train_dataset.val_size, val_correct, train_dataset.val_size, float(val_correct)/train_dataset.val_size)
		)

	return model

def test(model_t):
	test_data = one_hot_encode(get_data(test_dataset.data), get_unique())
	test_out = model_t.forward(test_data)[:,-1,:]
	test_pred = torch.argmax(test_out,1).reshape(-1,1)

	with open("pred.csv", "w+") as file:
		file.write("id,label\n\n")
		for i in range(test_pred.shape[0]):
			file.write("%d,%d\n"%(i,test_pred[i]))

if __name__ == "__main__":
	model_t = train()
	test(model_t)
