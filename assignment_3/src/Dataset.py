import torchfile
from PIL import Image
import sys

def disp_image(images, idx):
	img = Image.fromarray(images[idx])
	img = img.resize((256, 256))
	img.show()

data_train = torchfile.load('../data/dataset/train/data.bin')
disp_image(data_train, int(sys.argv[1]))