from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import theano 
import theano.tensor as T

from cnn import LeNetConvPoolLayer

b_size = 2
def generate_patch(img, (i, j), p_width = 10): # patch area : (10*10)
	assert p_width%2 == 0
	p_width/=2
	patch = img.crop((i-p_width, j-p_width, i+p_width, j+p_width))
	return patch

img = Image.open(open('../data/left_scaled/0080_left.jpg'))
print img.size
#print np.array(img).shape
patch = generate_patch(img, (0, 0))
patch2 = generate_patch(img, (2, 0))
#print np.array(patch).shape, np.array(patch)
patch1_ = np.array(patch, dtype='float64').transpose(2, 0, 1)/256.
patch2_ = np.array(patch2, dtype='float64').transpose(2, 0, 1)/256.
x = np.concatenate((patch1_, patch2_)).reshape((b_size, 3, 10, 10))

plt.imshow(patch)
plt.show()

label = Image.open(open('../data/labeled_scaled/0080_left.png'))
label = np.array(label, dtype='int32')
print 'label:', label.shape
print label[20][140]

rng = np.random.RandomState(23455)
print x.shape
#layer0_input = patch1_.transpose(2, 0, 1).reshape((1, 3, 10, 10))
#layer0_input = patch1_.reshape((1, 3, 10, 10))
layer0_input = x[1].reshape((1, 3, 10, 10))
#print layer0_input.shape
print 'equal: ', np.array_equal(x, x[0:2])

#print 'patch1_ dat:', patch_[0]

#added
#input = T.tensor4(name='input')
#w_shp = ()
#added

layer0 = LeNetConvPoolLayer(
	rng, 
	input = layer0_input,
	image_shape=(1, 3, 10, 10),
	filter_shape=(5, 3, 3, 3),
	poolsize=(2, 2)
)

layer1 = LeNetConvPoolLayer(
	rng, 
	input = layer0.output,
	image_shape=(1, 5, 4, 4),
	filter_shape=(8, 5, 2, 2),
	poolsize=(1, 1)
)
'''
print layer0.input.shape, layer0.output.shape.eval(), type(layer0.output.eval())
print layer0.input[0, 0, :,:]*256.
print layer0.output[0, 0, :, :].eval()*256.
print layer0.output[0, 1, :, :].eval()*256.'''
for i in range(5):
	plt.subplot(1, 5, i+1); plt.axis('off'); plt.imshow(Image.fromarray(layer0.output.eval()[0, i, :, :]*256.))
#plt.subplot(1, 6, 6); plt.axis('off'); plt.imshow(Image.fromarray(layer0.input[0, 0, :, :]*256.))
plt.show()

for i in range(8):
	plt.subplot(1, 8, i+1); plt.axis('off'); plt.imshow(Image.fromarray(layer1.output.eval()[0, i, :, :]*256.))
#plt.subplot(1, 8, 9); plt.axis('off'); plt.imshow(Image.fromarray(layer0.input[0, 0, :, :]*256.))
plt.show()
