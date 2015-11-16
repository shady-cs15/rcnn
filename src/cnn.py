import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import numpy
import timeit
import sys
import os

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


def trainConvNet(data_xy, n_epochs = 1, nkerns=[10, 20], batch_size=100, learning_rate=0.01):
	train_x, train_y, test_x, test_y, valid_x, valid_y = data_xy
	print train_x.shape.eval(), train_y.shape.eval()
	n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_x.get_value(borrow=True).shape[0] / batch_size
	print '...building the model'

	index = T.lscalar()

	x = T.matrix('x')
	y = T.ivector('y')
	#y = T.matrix('y')
	rng = numpy.random.RandomState(23455)

	layer0_input = x.reshape((batch_size, 3, 13, 13))

	layer0 = LeNetConvPoolLayer(
		rng, 
		input = layer0_input,
		image_shape=(batch_size, 3, 13, 13),
		filter_shape=(nkerns[0], 3, 4, 4),
		poolsize=(2, 2)
	)

	layer1 = LeNetConvPoolLayer(
		rng,
		input = layer0.output,
		image_shape=(batch_size, nkerns[0], 5, 5),
		filter_shape=(nkerns[1], nkerns[0], 2, 2),
		poolsize=(2, 2)
	)

	layer2_input = layer1.output.flatten(2)

	layer2 = HiddenLayer(
		rng,
		input=layer2_input,
		n_in=nkerns[1]*2*2,
		n_out=500,
		activation=T.tanh
	)

	layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

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
	patience_increase = 2  # wait this much longer when a new best is
	improvement_threshold = 0.995  # a relative improvement of this much is
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

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
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