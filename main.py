import tensorflow.compat.v1 as tf 
import pandas as pd 
import numpy as np 
from DataReader import *
from MLP import *
from utils import *
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

# create  computation graph
with open('./data_set/words_idfs.txt') as f: 
	vocab_size = len(f.read().splitlines()) 
	mlp = MLP( vocab_size=vocab_size, hidden_size=100, num_class=20) 
	predicted_labels, loss = mlp.build_graph() 
	train_op = mlp.trainer(loss=loss, learning_rate=0.1) 
	train_data_reader, test_data_reader = load_dataset(batch_size = 128, vocab_size = vocab_size)
# open a session to run and write all parameter to txt file
with tf.Session() as sess:
	step, MAX_STEP = 0, 1000
	sess.run(tf.global_variables_initializer()) 
	while step < MAX_STEP: 
		train_data, train_labels = train_data_reader.next_batch() 
		plabels_eval, loss_eval, _ = sess.run( 
			[predicted_labels, loss, train_op], 
			feed_dict={ 
				mlp._X: train_data, 
				mlp._real_Y: train_labels
				}
			)
		trainable_variables = tf.trainable_variables() # tf.trainable_variables: contains variables 
		
		step += 1 
		if step % 100 == 0:
			print('step: {}, loss: {}'.format(step, loss_eval))
			print('num_epoch: {}'.format(train_data_reader._num_epoch))
	for variable in trainable_variables:
		save_parameters(
			name = variable.name,
			value = variable.eval(),
			epoch = train_data_reader._num_epoch
		)

# load parameter from txt file and use MLP to eval on test data

with tf.Session() as sess:
	epoch = train_data_reader._num_epoch 
	for variable in trainable_variables:
		# print(variable.name)
		saved_value = restore_parameters(variable.name, epoch) 
		assign_op = variable.assign(saved_value) 
		sess.run(assign_op)

	num_true_preds = 0
	while True: 
		test_data, test_labels = test_data_reader.next_batch() 
		test_plabels_eval = sess.run(
									predicted_labels, 
									feed_dict={ 
										mlp._X: test_data, 
										mlp._real_Y: test_labels
										} 
									)
		matches = np.equal(test_plabels_eval, test_labels)
		num_true_preds += np.sum(matches.astype(float))
		if test_data_reader._batch_id == 0: 
			break
	print('Epoch:', epoch)
	print('Accuracy on test data:', num_true_preds / len(test_data_reader._data)) 
	# Classification report and confusion matrix
	# labels = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 
	#           'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 
	#           'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 
	#           'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
	print(classification_report(test_labels, test_plabels_eval))
	plt.rcParams["figure.figsize"] = (10, 20)
	plot_confusion_matrix(confusion_matrix(test_labels, test_plabels_eval),
						  show_normed=True,
						  show_absolute=False)
						  # class_names=labels)
	plt.show()