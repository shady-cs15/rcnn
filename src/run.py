from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

import theano 
import theano.tensor as T

import os

from cnn import trainConvNet

def generate_data(file_prefixes):
	def generate_patch(img, (i, j), p_width = 10): # patch area : (10*10)
		assert p_width%2 == 0
		p_width/=2
		patch = img.crop((i-p_width, j-p_width, i+p_width, j+p_width))
		return patch

	#load images
	patch_tuple = ()
	y_list = []
	data_size = 0
	ind = 0
	print 'Data generated: 0.00 %'
	for f in file_prefixes:
		img = Image.open(open('../data/left_scaled/'+f+'.jpg'))
		label = Image.open(open('../data/labeled_scaled/'+f+'.png'))
		label = np.array(label, dtype='int32')
		rows, cols = img.size
		for i in range(rows):
			for j in range(cols):
				# note img(i, j) -> label(j, i)
				l = label[j][i]
				if l==0:
					continue
				patch_10 = generate_patch(img, (i, j))
				patch_10_ = np.array(patch_10, dtype='float64').transpose(2, 0, 1)/256.
				patch_tuple += (patch_10_, )
				y_list.append(l-1)
				data_size += 1
				#print data_size
		ind += 1
		print '\033[FData generated: ', ind*100./len(file_prefixes), ' %\t'

	assert len(patch_tuple) == len(y_list) and len(y_list) == data_size
	print 'data size: ', data_size

	x = np.concatenate(patch_tuple).reshape((data_size, 3, 10, 10))
	y = y_list

	return x, y

def shared_dataset(data_xy):
	x, y = data_xy
	shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
	shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=True)
	return shared_x, T.cast(shared_y, 'int32')


'''
	program starts here
'''
file_prefixes = []
for f in os.listdir('../data/labeled_scaled'):
	if f=='.DS_Store':
		continue
	file_prefixes.append(f[:-4])

#shuffle(file_prefixes)

x, y = generate_data(file_prefixes[:4])
train_x, train_y = shared_dataset((x[0:(3*len(x)/5)], y[0:(3*len(y)/5)]))
valid_x, valid_y = shared_dataset((x[(3*len(x)/5):(4*len(x)/5)], y[3*len(y)/5:4*len(y)]))
test_x, test_y = shared_dataset((x[(4*len(x)/5):len(x)], y[(4*len(y)/5):len(y)]))

print type(test_x), type(test_y)
trainConvNet((train_x, train_y, test_x, test_y, valid_x, valid_y))
