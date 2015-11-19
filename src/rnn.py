import theano
import theano.tensor as T

import numpy

from mlp import HiddenLayer2

def trainRecNet(data_xy, inp_dim = 90, n_epochs = 5, batch_size=500, learning_rate=0.1, n_recurrences=4):
	train_x, train_y, test_x, test_y, valid_x, valid_y = data_xy
	# important train_x = (train_x0, train_x1, train_x2, train_x3)

	n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_x.get_value(borrow=True).shape[0] / batch_size
	print '...building the model'
	
	index = T.lscalar()

	y = T.ivector('y')
	rng = numpy.random.RandomState(23455)

	# initialize layer1
	# layer1 = ... HiddenLayer2 ...
	for i in range(n_recurrences):
		x = T.imatrix('x')
		layer0 = x.reshape((batch_size, inp_dim)).flatten(2)

		layer_1 = layer1.output

		layer1 = HiddenLayer2(
			rng,
			input0=layer0,
			input_1=layer_1,
			n_in0=inp_dim,
			n_in_1=300,
			n_out=300,
			activation=T.tanh,
			U=layer_1.U,
			W=layer_1.W,
			b=layer_1.b
		)

		layer2 = LogisticRegression(input=layer1.output, n_in=300, n_out=10)

		cost = layer2.negative_log_likelihood(y)

		params = layer2.params + layer1.params

		grads = T.grad(cost, params)

		updates = [
			(param_i, param_i - learning_rate * grad_i)
			for param_i, grad_i in zip(params, grads)
		] 

		train_model = theano.function([index], cost, updates=updates, givens={
			x: train_x[i][index*batch_size: (index+1)*batch_size],
			y: train_y[index*batch_size: (index+1)*batch_size]
		})

		# train using function train_model inside the loop