from utils import *
from Model import Model

train_data = Dataset("../data/train_data.txt",3)
print(train_data.size)
print(train_data.batch_size)
print(train_data.num_batches)
print(train_data.get_batch(0).shape)

# rnn = Model(2, 16, 266, 2)
# print(rnn.forward(train_data.get_batch(0), torch.zeros(3,2,16)).shape)
# print(rnn.outputs.shape)