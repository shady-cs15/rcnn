import theano
import theano.tensor as T

import numpy
import timeit

from mlp import HiddenLayer, HiddenLayer2
from logistic_sgd import LogisticRegression

def trainRecNet(data_xy, inp_dim = 90, n_epochs = 5, batch_size=500, learning_rate=0.1, n_recurrences=4):
	train_x, train_y, test_x, test_y, valid_x, valid_y = data_xy
	# important train_x = (train_x0, train_x1, train_x2, train_x3)
	# so is test_x, valid_x

	print type(train_x[0]), train_x[0].shape.eval()
	n_train_batches = train_x[0].get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_x[0].get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_x[0].get_value(borrow=True).shape[0] / batch_size
	print '...building the model'
	print 'n_train_batches: ', n_train_batches
	index = T.lscalar()

	rng = numpy.random.RandomState(23455)

	for i in range(n_recurrences):
		'''x = T.dmatrix('x')
		y = T.ivector('y')'''

		x = train_x[i][index*batch_size: (index+1)*batch_size]
		y = train_y[index*batch_size: (index+1)*batch_size]

		layer0 = x

		if i == 0:
			layer1 = HiddenLayer(
				rng,
				input=layer0,
				n_in=inp_dim,
				n_out=300,
				activation=T.tanh
			)
				
		if i==1:
			layer_1_output = layer1.output.flatten(2)
			layer_1 = layer1

			layer1 = HiddenLayer2(
				rng,
				input0=layer0,
				input_1=layer_1_output,
				n_in0=inp_dim,
				n_in_1=300,
				n_out=300,
				U=layer_1.W,
				W=None,
				b=layer_1.b,
				activation=T.tanh
			)

		if i > 1:
			layer_1_output = layer1.output.flatten(2)
			layer_1 = layer1

			layer1 = HiddenLayer2(
				rng,
				input0=layer0,
				input_1=layer_1_output,
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

		train_model = theano.function([index], cost, updates=updates)#''', givens={
		#	x: train_x[i][index*batch_size: (index+1)*batch_size],
		#	y: train_y[index*batch_size: (index+1)*batch_size]
		#})'''

		# train using function train_model inside the loop
		print '...training at recurrence step: ', i
		epoch = 0
		done_looping = False
		patience = 10000
		patience_increase = 2
		improvement_threshold = 0.995
		validation_frequency = min(n_train_batches, patience/2)
		best_validation_loss = numpy.inf
		best_iter = 0
		test_score = 0.0
		start_time = timeit.default_timer()

		while((epoch < n_epochs) and (not done_looping)):
			epoch += 1
			for minibatch_index in xrange(n_train_batches):
				iter = (epoch-1) * n_train_batches + minibatch_index
				if iter % 100 == 0:
					print 'training @ iter =', iter
				cost_ij = train_model(minibatch_index)

	print 'layer 1 params: ', layer1.W.shape.eval(), layer1.U.shape.eval(), layer1.b.shape.eval()
	print 'layer 2 params: ', layer2.W.shape.eval(), layer2.b.shape.eval()