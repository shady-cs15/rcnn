import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import numpy
import timeit
import sys
import os
import cPickle

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), read_file=False, W_input=None, b_input=None):
        
        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        if read_file==True:
        	self.W = W_input
        	self.b = b_input

        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        self.params = [self.W, self.b]

        self.input = input

def represent(data_x, batch_size, patch_width=10, nkerns=[5, 10]):

	kern0_dim = 3
	kern1_dim = 2
	pool0_dim = 2
	pool1_dim = 1

	if patch_width==4:
		kern0_dim = 2
		kern1_dim = 1
		pool0_dim = 1
		pool1_dim = 1

	if patch_width==10:
		kern0_dim = 3
		kern1_dim = 2
		pool0_dim = 2
		pool1_dim = 1

	if patch_width==14:
		kern0_dim = 5
		kern1_dim = 3
		pool0_dim = 2
		pool1_dim = 1

	if patch_width==20:
		kern0_dim = 7
		kern1_dim = 5
		pool0_dim = 2
		pool1_dim = 1

	rng = numpy.random.RandomState(23455)
	W0 = theano.shared(numpy.asarray(rng.uniform(low=-1., high=-1., size=(nkerns[0], 3, kern0_dim, kern0_dim)), dtype=theano.config.floatX), borrow=True)
	b0 = theano.shared(numpy.asarray(rng.uniform(low=-1., high=-1., size=(nkerns[0],)), dtype=theano.config.floatX), borrow=True)
	W1 = theano.shared(numpy.asarray(rng.uniform(low=-1., high=-1., size=(nkerns[1], nkerns[0], kern1_dim, kern1_dim)), dtype=theano.config.floatX), borrow=True)
	b1 = theano.shared(numpy.asarray(rng.uniform(low=-1., high=-1., size=(nkerns[1],)), dtype=theano.config.floatX), borrow=True)
	save_file = open('param'+str(patch_width)+'.pkl')
	W0.set_value(cPickle.load(save_file), borrow=True)
	b0.set_value(cPickle.load(save_file), borrow=True)
	W1.set_value(cPickle.load(save_file), borrow=True)
	b1.set_value(cPickle.load(save_file), borrow=True)
	save_file.close()

	inp_dim = patch_width

	layer0_input = data_x.reshape((batch_size, 3, inp_dim, inp_dim))

	layer0 = LeNetConvPoolLayer(
		rng, 
		input = layer0_input,
		image_shape=(batch_size, 3, inp_dim, inp_dim),
		filter_shape=(nkerns[0], 3, kern0_dim, kern0_dim),
		poolsize=(pool0_dim, pool0_dim),
		read_file=True,
		W_input=W0,
		b_input=b0
	)

	
	inp1_dim = (inp_dim-kern0_dim+1)/pool0_dim
	layer1 = LeNetConvPoolLayer(
		rng,
		input = layer0.output,
		image_shape=(batch_size, nkerns[0], inp1_dim, inp1_dim),
		filter_shape=(nkerns[1], nkerns[0], kern1_dim, kern1_dim),
		poolsize=(pool1_dim, pool1_dim),
		read_file=True,
		W_input=W1,
		b_input=b1
	)

	layer2_input = layer1.output.flatten(2)

	# compile into a theano function
	f = theano.function([data_x], layer2_input)
	return f(data_x.eval())

def trainConvNet(data_xy, inp_dim =10, n_epochs = 3, nkerns=[5, 10], batch_size=500, learning_rate=0.1):
	train_x, train_y, test_x, test_y, valid_x, valid_y = data_xy

	n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_x.get_value(borrow=True).shape[0] / batch_size
	print '...building the model'

	kern0_dim = 3
	kern1_dim = 2
	pool0_dim = 2
	pool1_dim = 1

	if inp_dim==4:
		kern0_dim = 2
		kern1_dim = 1
		pool0_dim = 1
		pool1_dim = 1

	if inp_dim==10:
		kern0_dim = 3
		kern1_dim = 2
		pool0_dim = 2
		pool1_dim = 1

	if inp_dim==14:
		kern0_dim = 5
		kern1_dim = 3
		pool0_dim = 2
		pool1_dim = 1

	if inp_dim==20:
		kern0_dim = 7
		kern1_dim = 5
		pool0_dim = 2
		pool1_dim = 1

	index = T.lscalar()

	x = T.tensor4('x')
	y = T.ivector('y')
	rng = numpy.random.RandomState(23455)

	layer0_input = x.reshape((batch_size, 3, inp_dim, inp_dim))

	layer0 = LeNetConvPoolLayer(
		rng, 
		input = layer0_input,
		image_shape=(batch_size, 3, inp_dim, inp_dim),
		filter_shape=(nkerns[0], 3, kern0_dim, kern0_dim),
		poolsize=(pool0_dim, pool0_dim)
	)

	inp1_dim = (inp_dim-kern0_dim+1)/pool0_dim
	layer1 = LeNetConvPoolLayer(
		rng,
		input = layer0.output,
		image_shape=(batch_size, nkerns[0], inp1_dim, inp1_dim),
		filter_shape=(nkerns[1], nkerns[0], kern1_dim, kern1_dim),
		poolsize=(pool1_dim, pool1_dim)
	)

	layer2_input = layer1.output.flatten(2)

	inp2_dim = (inp1_dim-kern1_dim+1)/pool1_dim
	layer2 = HiddenLayer(
		rng,
		input=layer2_input,
		n_in=nkerns[1]*inp2_dim*inp2_dim,
		n_out=300,
		activation=T.tanh
	)

	layer3 = LogisticRegression(input=layer2.output, n_in=300, n_out=10)

	cost = layer3.negative_log_likelihood(y)


	test_model = theano.function([index], layer3.errors(y), givens={
			x: test_x[index*batch_size: (index+1)*batch_size],
			y: test_y[index*batch_size: (index+1)*batch_size]
		})

	validate_model = theano.function([index], layer3.errors(y), givens={
			x: valid_x[index*batch_size: (index+1)*batch_size],
			y: valid_y[index*batch_size: (index+1)*batch_size]
		})

	params = layer3.params + layer2.params + layer1.params + layer0.params

	grads  = T.grad(cost, params)

	updates = [
		(param_i, param_i - learning_rate * grad_i)
		for param_i, grad_i in zip(params, grads)
	] 

	train_model = theano.function([index], cost, updates=updates, givens={
			x: train_x[index*batch_size: (index+1)*batch_size],
			y: train_y[index*batch_size: (index+1)*batch_size]
		})

	print 'training... '

	patience = 10000
	patience_increase = 2
	improvement_threshold = 0.995
	validation_frequency = min(n_train_batches, patience / 2)
	best_validation_loss = numpy.inf
	best_iter = 0
	test_score = 0.
	start_time = timeit.default_timer()

	epoch = 0
	done_looping = False

	while (epoch < n_epochs) and (not done_looping):
		epoch = epoch + 1
		for minibatch_index in xrange(n_train_batches):
			iter = (epoch - 1) * n_train_batches + minibatch_index
			if iter % 100 == 0:
				print 'training @ iter = ', iter
			cost_ij = train_model(minibatch_index)

			if (iter + 1) % validation_frequency == 0:
				validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
				this_validation_loss = numpy.mean(validation_losses)
				print('epoch %i, minibatch %i/%i, validation error %f %%\n' %(epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:

                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of ' 'best model %f %%') %(epoch, minibatch_index + 1, n_train_batches, test_score * 100.))
          	if patience<=iter:
          		done_looping=True
          		break

	end_time = timeit.default_timer()
	print('Optimization complete.')
	print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
	print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

	print ('saving params for patch width: %i...' %(inp_dim))
	save_file = open('param'+str(inp_dim)+'.pkl', 'wb')
	W0 = layer0.params[0]; b0 = layer0.params[1]
	W1 = layer1.params[0]; b1 = layer1.params[1]
	cPickle.dump(W0.get_value(borrow=True), save_file, -1)
	cPickle.dump(b0.get_value(borrow=True), save_file, -1)
	cPickle.dump(W1.get_value(borrow=True), save_file, -1)
	cPickle.dump(b1.get_value(borrow=True), save_file, -1)
	save_file.close()
