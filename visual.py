import pickle
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
	hist_in = open('histories/training_history.pickle', 'rb')
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
	plt.subplot(2, 1, 1)
	plt.plot(epochs, train_sensitivity)
	plt.plot(epochs, val_sensitivity)
	plt.title('Sensitivity Curve')
	plt.xlabel('Epochs')
	plt.ylabel('Sensitivity')
	plt.gca().legend(('Training', 'Validation'))

	plt.subplot(2, 1, 2)
	plt.tight_layout(4)
	plt.plot(epochs, train_specificity)
	plt.plot(epochs, val_specificity)
	plt.title('Specificity Curve')
	plt.xlabel('Epochs')
	plt.ylabel('Specificity')
	plt.gca().legend(('Training', 'Validation'))

	plt.show()
