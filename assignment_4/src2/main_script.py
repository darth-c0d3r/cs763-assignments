from utils import *
from Model import Model

train_data = Dataset("../data/train_data.txt",3)
print(train_data.size)
print(train_data.batch_size)
print(train_data.num_batches)
print(train_data.get_batch(0).shape)
print(type(train_data.get_batch(0)))

model = Model(2, 16, 154, 2)
y = model.forward(train_data.get_batch(0))
x = model.backward(y)
print(y.shape)
print(x.shape)
# print(rnn.outputs.shape)