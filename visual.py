import pickle
import seaborn
import numpy as np
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import roc_curve, auc


def sensitivity(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
	true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
	possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
	return true_negatives / (possible_negatives + K.epsilon())


if __name__ == '__main__':
	hist_in = open('histories/training_history_100.pickle', 'rb')
	history = pickle.load(hist_in)
	hist_in.close()
	train_loss = np.array(history['loss'])
	val_loss = np.array(history['val_loss'])
	train_acc = np.array(history['acc'])
	val_acc = np.array(history['val_acc'])
	train_sensitivity = np.array(history['sensitivity'])
	val_sensitivity = np.array(history['val_sensitivity'])
	train_specificity = np.array(history['specificity'])
	val_specificity = np.array(history['val_specificity'])
	train_auc = np.array(history['auc'])
	val_auc = np.array(history['val_auc'])
	epochs = np.arange(1, train_acc.shape[0]+1)

	f1 = plt.figure(1)
	plt.style.use('seaborn')
	plt.subplot(2, 1, 1)
	plt.plot(epochs, train_acc*100)
	plt.plot(epochs, val_acc*100)
	plt.title('Accuracy Curve')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy%')
	plt.gca().legend(('Training', 'Validation'))

	plt.subplot(2, 1, 2)
	plt.tight_layout(4)
	plt.plot(epochs, train_loss)
	plt.plot(epochs, val_loss)
	plt.title('Loss Curve')
	plt.xlabel('Epochs')
	plt.ylabel('Loss%')
	plt.gca().legend(('Training', 'Validation'))

	f2 = plt.figure(2)
	plt.style.use('seaborn')
	plt.subplot(3, 1, 1)
	plt.plot(epochs, train_sensitivity)
	plt.plot(epochs, val_sensitivity)
	plt.title('Sensitivity Curve')
	plt.xlabel('Epochs')
	plt.ylabel('Sensitivity')
	plt.gca().legend(('Training', 'Validation'))

	plt.subplot(3, 1, 2)
	plt.tight_layout(0)
	plt.plot(epochs, train_specificity)
	plt.plot(epochs, val_specificity)
	plt.title('Specificity Curve')
	plt.xlabel('Epochs')
	plt.ylabel('Specificity')
	plt.gca().legend(('Training', 'Validation'))

	plt.subplot(3, 1, 3)
	plt.tight_layout(0)
	plt.plot(epochs, train_auc)
	plt.plot(epochs, val_auc)
	plt.title('AUC Curve')
	plt.xlabel('Epochs')
	plt.ylabel('AUC')
	plt.gca().legend(('Training', 'Validation'))

	f3 = plt.figure(3)
	plt.style.use('seaborn')
	plt.subplot(2, 1, 1)
	model = load_model('models/rnnmodel.h5', custom_objects={'sensitivity': sensitivity, 'specificity': specificity}, compile=True)
	X_test_in, y_test_in = open('testdata/X_test_100.pickle', 'rb'), open('testdata/y_test_100.pickle', 'rb')
	X_test, y_test = pickle.load(X_test_in), pickle.load(y_test_in)
	X_test_in.close()
	y_test_in.close()
	y_pred = np.round(model.predict(X_test))
	matrix = tf.math.confusion_matrix(y_test, y_pred)
	matrix = matrix/matrix.numpy().sum(axis=1)[:, tf.newaxis]	
	seaborn.heatmap(matrix, annot=True)
	plt.title('Confusion Matrix')
	plt.xlabel("Predicted")
	plt.ylabel("True")

	plt.subplot(2, 1, 2)
	plt.tight_layout(4)
	fp, tp, thresholds = roc_curve(y_test, y_pred)
	auc = auc(fp, tp)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fp, tp, label='Area = {:.3f}'.format(auc))
	plt.xlabel('False positive')
	plt.ylabel('True positive')
	plt.title('ROC curve')
	plt.legend(loc='best')

	plt.show()
