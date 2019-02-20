import torch
import torchfile
from numpy import random
import scipy.ndimage as ndimage
import numpy as np

def disp_image(images, idx):
	from PIL import Image
	img = Image.fromarray(images[idx])
	img = img.resize((256, 256))
	img.show()

def normalize_data(data):

	data = data / torch.max(data)
	data = data - torch.mean(data)
	return data

def get_device(cuda = 1):
	device = torch.device("cuda" if torch.cuda.is_available() and cuda == 1 else "cpu")
	print("Using Device:", device)
	return device

def rotate_img(img, angle):

	def crop_center(img_,cropx,cropy):
	    y,x = img_.shape
	    startx = x//2-(cropx//2)
	    starty = y//2-(cropy//2)    
	    return img_[starty:starty+cropy,startx:startx+cropx]

	img_rot = crop_center(ndimage.rotate(img, angle), img.shape[0], img.shape[1])
	return img_rot

def augment_data(input_data, input_labels):

	print("Augmenting Data")
	output_data = np.zeros((input_data.shape[0]*3, input_data.shape[1], input_data.shape[2]))
	output_labels = np.zeros((input_labels.shape[0]*3))
	for i in range(len(input_data)):
		output_data[3*i] = input_data[i]
		output_data[3*i+1] = np.flip(input_data[i], 0)
		output_data[3*i+2] = np.flip(input_data[i], 1)
		output_labels[3*i] = input_labels[i]
		output_labels[3*i+1] = input_labels[i]
		output_labels[3*i+2] = input_labels[i]
	print("Augmentation Done")
	return output_data, output_labels

def get_data(data_path, label_path, device, augment=False):

	input_data = torchfile.load(data_path)
	input_labels = torchfile.load(label_path)

	if augment is True:
		input_data, input_labels = augment_data(input_data, input_labels)

	input_data = torch.tensor(input_data.reshape(input_data.shape[0],-1)).double().to(device)
	input_labels = torch.tensor(input_labels.reshape(input_labels.shape[0],-1)).long().to(device)

	return input_data, input_labels

def split_data(input_data, input_labels):
	train_data = input_data[:int(0.9 * input_data.shape[0]),:]
	train_labels = input_labels[:int(0.9 * input_labels.shape[0]),:]

	# train_data = input_data[:500,:]
	# train_labels = input_labels[:500,:]

	val_data = input_data[int(0.9 * input_data.shape[0]):,:]
	val_labels = input_labels[int(0.9 * input_labels.shape[0]):,:]

	print("Training Set Size: %d" % (train_data.shape[0]))
	print("Validation Set Size: %d" % (val_data.shape[0]))
	print("Input Vector Size: %d" % (train_data.shape[1]))

	return train_data, train_labels, val_data, val_labels

def shuffle(data, labels):
	a = [i for i in range(data.shape[0])]
	random.shuffle(a)

	data_sh, labels_sh = data[a], labels[a]

	torch.cuda.empty_cache()

	return data_sh, labels_sh