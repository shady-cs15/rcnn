import theano
import theano.tensor as T

import numpy
import timeit
import cPickle

from mlp import HiddenLayer, HiddenLayer2
from logistic_sgd import LogisticRegression

def trainRecNet(data_xy, inp_dim = 90, n_epochs = 5, batch_size=500, learning_rate=0.1, n_recurrences=4):
	train_x, train_y = data_xy
	n_train_batches = train_x[0].get_value(borrow=True).shape[0] / batch_size
	print '...building the RNN model'
	index = T.lscalar()

	rng = numpy.random.RandomState(23455)

	for i in range(n_recurrences):
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

		train_model = theano.function([index], cost, updates=updates)

		print '...training at recurrence step: ', i
		epoch = 0
		start_time = timeit.default_timer()

		print '\n'
		while((epoch < n_epochs)):
			epoch += 1
			for minibatch_index in xrange(n_train_batches):
				iter = (epoch-1) * n_train_batches + minibatch_index
				if iter % 100 == 0:
					print '\033[Ftraining @ iter =', iter
				cost_ij = train_model(minibatch_index)

	save_file = open('rnnparams.pkl', 'wb')
	cPickle.dump(layer1.W.get_value(borrow=True), save_file, -1)
	cPickle.dump(layer1.U.get_value(borrow=True), save_file, -1)
	cPickle.dump(layer1.b.get_value(borrow=True), save_file, -1)
	cPickle.dump(layer2.W.get_value(borrow=True), save_file, -1)
	cPickle.dump(layer2.b.get_value(borrow=True), save_file, -1)
	save_file.close()
	
def evaluate(data_xy, inp_dim=90, batch_size=500, n_recurrences=4):
	test_x, test_y = data_xy
	n_test_batches = test_x[0].get_value(borrow=True).shape[0] / batch_size

	rng = numpy.random.RandomState(23455)
	
	W1 = theano.shared(numpy.asarray(rng.uniform(low=-1., high=-1., size=(300, 300)), dtype=theano.config.floatX), borrow=True)
	U1 = theano.shared(numpy.asarray(rng.uniform(low=-1., high=-1., size=(90, 300)), dtype=theano.config.floatX), borrow=True)
	b1 = theano.shared(numpy.asarray(rng.uniform(low=-1., high=-1., size=(300,)), dtype=theano.config.floatX), borrow=True)
	W2 = theano.shared(numpy.asarray(rng.uniform(low=-1., high=-1., size=(300, 10)), dtype=theano.config.floatX), borrow=True)
	b2 = theano.shared(numpy.asarray(rng.uniform(low=-1., high=-1., size=(10, )), dtype=theano.config.floatX), borrow=True)
	

	save_file = open('rnnparams.pkl')
	W1.set_value(cPickle.load(save_file), borrow=True)
	U1.set_value(cPickle.load(save_file), borrow=True)
	b1.set_value(cPickle.load(save_file), borrow=True)
	W2.set_value(cPickle.load(save_file), borrow=True)
	b2.set_value(cPickle.load(save_file), borrow=True)
	save_file.close()

	index = T.lscalar()

	# start of theano function
	x = test_x[0][index*batch_size: (index+1)*batch_size]
	y = test_y[index*batch_size: (index+1)*batch_size]

	layer0 = HiddenLayer(
		rng,
		input = x,
		n_in = 90,
		n_out = 300,
		W = U1,
		b = b1,
		activation = T.tanh
	)

	for i in range(1, n_recurrences):
		if i==1:
			inp = layer0.output
		else:
			inp = layer1.output

		layer1 = HiddenLayer2(
			rng,
			input0 = test_x[i][index*batch_size: (index+1)*batch_size],
			input_1 = inp,
			n_in0 = 90,
			n_in_1 = 300,
			n_out = 300,
			W = W1,
			U = U1,
			b = b1,
			activation = T.tanh
		)

	layer2 = LogisticRegression(input=layer1.output, n_in=300, n_out=10, W=W2, b=b2)
	
	cost = layer2.negative_log_likelihood(y)
	
	f = theano.function([index], cost)
	
	y_ = layer2.y_pred
	
	g = theano.function([index], y_)

	losses = [
		f(ind)
		for ind in xrange(n_test_batches)
	]
	score = numpy.mean(losses)

	preds = [
		g(ind)
		for ind in xrange(n_test_batches)
	]

	return preds
