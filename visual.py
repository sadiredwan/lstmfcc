import pickle
import seaborn
import numpy as np
import matplotlib.pyplot as pl
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import roc_curve, auc, confusion_matrix


def sensitivity(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
	true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
	possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
	return true_negatives / (possible_negatives + K.epsilon())


if __name__ == '__main__':
	hist_in = open('histories/training_history_05.pickle', 'rb')
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
	epochs = np.arange(1, train_acc.shape[0]+1)

	f1 = pl.figure(1)
	pl.style.use('seaborn')
	pl.subplot(2, 1, 1)
	pl.plot(epochs, train_acc*100)
	pl.plot(epochs, val_acc*100)
	pl.title('Accuracy Curve')
	pl.xlabel('Epochs')
	pl.ylabel('Accuracy%')
	pl.gca().legend(('Training', 'Validation'))

	pl.subplot(2, 1, 2)
	pl.tight_layout(4)
	pl.plot(epochs, train_loss)
	pl.plot(epochs, val_loss)
	pl.title('Loss Curve')
	pl.xlabel('Epochs')
	pl.ylabel('Loss%')
	pl.gca().legend(('Training', 'Validation'))

	f2 = pl.figure(2)
	pl.style.use('seaborn')
	pl.subplot(2, 1, 1)
	pl.plot(epochs, train_sensitivity)
	pl.plot(epochs, val_sensitivity)
	pl.title('Sensitivity Curve')
	pl.xlabel('Epochs')
	pl.ylabel('Sensitivity')
	pl.gca().legend(('Training', 'Validation'))

	pl.subplot(2, 1, 2)
	pl.tight_layout(4)
	pl.plot(epochs, train_specificity)
	pl.plot(epochs, val_specificity)
	pl.title('Specificity Curve')
	pl.xlabel('Epochs')
	pl.ylabel('Specificity')
	pl.gca().legend(('Training', 'Validation'))

	f3 = pl.figure(3)
	pl.style.use('seaborn')
	pl.subplot(2, 1, 1)
	model = load_model('models/rnnmodel_05.h5', custom_objects={'sensitivity': sensitivity, 'specificity': specificity}, compile=True)
	X_test_in, y_test_in = open('testdata/X_test_05.pickle', 'rb'), open('testdata/y_test_05.pickle', 'rb')
	X_test, y_test = pickle.load(X_test_in), pickle.load(y_test_in)
	X_test_in.close()
	y_test_in.close()
	y_pred = np.round(model.predict(X_test))
	matrix = confusion_matrix(y_test, y_pred)
	matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
	seaborn.heatmap(matrix, annot=True)
	pl.title('Confusion Matrix')
	pl.xlabel("Predicted")
	pl.ylabel("True")

	pl.subplot(2, 1, 2)
	pl.tight_layout(4)
	fp, tp, thresholds = roc_curve(y_test, y_pred)
	auc = auc(fp, tp)
	pl.plot([0, 1], [0, 1], 'k--')
	pl.plot(fp, tp, label='Area={:.3f}'.format(auc))
	pl.xlabel('False positive')
	pl.ylabel('True positive')
	pl.title('ROC curve')
	pl.legend(loc='best')

	pl.show()
