import theano
import theano.tensor as T

import numpy
import timeit

from mlp import HiddenLayer, HiddenLayer2

def trainRecNet(data_xy, inp_dim = 90, n_epochs = 5, batch_size=500, learning_rate=0.1, n_recurrences=4):
	train_x, train_y, test_x, test_y, valid_x, valid_y = data_xy
	# important train_x = (train_x0, train_x1, train_x2, train_x3)
	# so is test_x, valid_x

	n_train_batches = train_x[0].get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_x[0].get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_x[0].get_value(borrow=True).shape[0] / batch_size
	print '...building the model'
	
	index = T.lscalar()

	y = T.ivector('y')
	rng = numpy.random.RandomState(23455)

	for i in range(n_recurrences):
		x = T.imatrix('x')
		layer0 = x.reshape((batch_size, inp_dim)).flatten(2)

		if i == 0:
			layer1 = HiddenLayer(
				rng,
				input=layer0,
				n_in=inp_dim,
				n_out=300,
				activation=T.tanh
			)
		
		if i==1:
			layer_1 = layer1.output.flatten(2)

			layer1 = HiddenLayer2(
				rng,
				input0=layer0,
				input_1=layer_1,
				n_in0=inp_dim,
				n_in_1=300,
				n_out=300,
				activation=T.tanh,
				U=layer_1.W,
				b=layer_1.b
			)

		if i > 1:
			layer_1 = layer1.output.flatten(2)

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

		valid_model = theano.function([index], layer2.errors(y), givens={
			x: valid_x[i][index*batch_size: (index+1)*batch_size],
			y: valid_y[index*batch_size: (index+1)*batch_size]
		})

		test_model = theano.function([index], layer2.errors(y), givens={
			x: test_x[i][index*batch_size: (index+1)*batch_size],
			y: test_y[index*batch_size: (index+1)*batch_size]
		})
		
		# train using function train_model inside the loop
		print '...training at recurrence step: ', i
		epoch = 0
		done_looping = False
		patience = 10000
		patience_increase = 2
		improvement_threshold = 0.995
		validation_frequency = min(n_train_batches, patches/2)
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

				if (iter+1)% validation_frequency == 0:
					validation_losses = [valid_model(i) for i in xrange(n_valid_batches)]
					this_validation_loss = numpy.mean(validation_losses)
					print('epoch %i, minibatch %i/%i, validation error %f %%\n' %(epoch, minibatch_index +1, n_train_batches, this_validation_loss * 100.))

					if this_validation_loss < best_validation_loss:
						if this_validation_loss < best_validation_loss * \
						improvemnet_threshold:
							patience = max(patience, iter * patience_increase)

						best_validation_loss = this_validation_loss
						best_iter = iter

						test_losses = [
							test_model(i)
							for i in xrange(n_test_batches)
						]

						test_score = numpy.mean(test_losses)
						print (('     epoch %i, minibatch %i/%i, test error of ' 'best model %f %%') %(epoch, minibatch_index+1, n_train_batches, test_score * 100.))

				if patience<=iter:
					done_looping = True
					break

		end_time = timeit.default_timer()
		print('First step of recurrence complete.')
		print('Best validation score of %f %% obtained at iteration %i, ' 'with test performance %f %%' %(best_validation_loss * 100., best_iter+1, test_score*100.))
		print('Step '+i+' ran for %.2fm' %((end_time - start_time)/ 60.))