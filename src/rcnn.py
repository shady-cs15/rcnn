import os
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import sys

import theano
import theano.tensor as T

import cPickle

from cnn import trainConvNet

bin_index = int(sys.argv[1])
bin_size = 5

print 'note that arg should be 0..9'

labeled_files = [f for f in os.listdir('../data/labeled_scaled')]
#labeled_files = labeled_files[bin_size*bin_index:bin_size*(bin_index+1)] 
labeled_files = labeled_files[1:4]
print "labeled files loaded: ", len(labeled_files)

#print labeled_files

data_files = []
for f in labeled_files:
	data_files.append(f[:-3]+'jpg')
	#print f[:-3]+'jpg'
#data_files = data_files[bin_size*bin_index:bin_size*(bin_index+1)]
print "data files loaded: ", len(data_files)

y_list = []
x_list = []
	
# loads dataset from jpg and png
# into train, cross_valid, test
def load_dataset(data_files, create_dump=False):
	global y_list, x_list
	if len(data_files)!=len(labeled_files):
		print "size mismatch between files"
		return
	
	print 'reading labeled files...'
	for f in labeled_files:
		label = misc.imread('../data/labeled_scaled/'+f)
		y = vectorize_labels(label)
		y_list+=y
	if create_dump==True:
		save_file = open('../generated/data_y'+str(bin_index), 'wb')
		cPickle.dump(y_list, save_file, -1)
		save_file.close()

	print 'reading data files...\n'	
	for f in data_files:
		img = misc.imread('../data/left_scaled/'+f)
		label_img = misc.imread('../data/labeled_scaled/'+f[:-3]+'png')
		patches = compute_patches(img, label_img, 6)
		x_list+=patches
		print '\033[Fdata loaded: ', len(x_list)*100.0/len(y_list), ' %'
		#print len(patches)
	if create_dump==True:
		print 'dumping data..'
		save_file = open('../generated/data_x'+str(bin_index), 'wb')
		cPickle.dump(x_list, save_file, -1)
		save_file.close()

	print 'size of dataset: ', len(y_list), len(x_list)
	return x_list, y_list

def vectorize_labels(img):
	rows, cols, channels = img.shape
	lst = []
	for i in range(rows):
		for j in range(cols):
			if img[i][j][0]==0:
				continue
			lst.append(img[i][j][0]-1)
	return lst 

def compute_patches(img, label_img, p_width = 4):   #patches are of size 2*p_width+1 ^ 2
	rows, cols, channels = img.shape
	patch_list = []
	for i in range(rows):
		for j in range(cols):
			if label_img[i][j][0]==0:
				continue
			temp_list = []
			for m in range(3):
				pixel = []
				for k in range(i-p_width, i+p_width+1):
					for l in range(j-p_width, j+p_width+1):
						if k < 0 or k >= rows or l < 0 or l >= cols:
							pixel+=[0]
						else:
							pixel+=[img[k][l][m]]
				temp_list+=(pixel)
			patch_list.append(temp_list)
	return patch_list

def map_to_onehot(num):
	assert num<11 and num > 0
	lst = []
	for i in range(10):
		lst.append(0)
	lst[num-1]=1
	return lst

# loads into shared memory
def shared_dataset(data_xy):
	x, y = data_xy
	shared_x = theano.shared(np.asarray(x, dtype=theano.config.floatX), borrow=True)
	shared_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=True)
	return shared_x, T.cast(shared_y, 'int32')

x, y = load_dataset(data_files, False)
print 'loading into shared memory...'
train_x, train_y = shared_dataset((x[0:(3*len(x)/5)], y[0:(3*len(x)/5)]))
valid_x, valid_y = shared_dataset((x[(3*len(x)/5):(4*len(x)/5)], y[3*len(x)/5:4*len(x)]))
test_x, test_y = shared_dataset((x[(4*len(x)/5):len(x)], y[(4*len(x)/5):len(x)]))
print 'sample data shape...', train_x.shape.eval(), train_y.shape.eval()
trainConvNet((train_x, train_y, test_x, test_y, valid_x, valid_y))
