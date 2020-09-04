import pickle
import seaborn
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.patches as pt


def autolabel(rects):
	for rect in rects:
		height = rect.get_height()
		ax.annotate('{:.4f}'.format(height),
			xy=(rect.get_x()+rect.get_width()/2, height),
			xytext=(0, 3),
			textcoords='offset points',
			ha='center', va='bottom')


def patch_handle():
	patches = []
	ax.get_children()[0].set_color(seaborn.color_palette('GnBu_d')[5])
	patches.append(pt.Patch(color=seaborn.color_palette('GnBu_d')[5], label='nfft=128'))
	for i in range(1, 4):
		ax.get_children()[i].set_color(seaborn.color_palette('GnBu_d')[4])
	patches.append(pt.Patch(color=seaborn.color_palette('GnBu_d')[4], label='nfft=512'))

	for i in range(4, 9):
		ax.get_children()[i].set_color(seaborn.color_palette('GnBu_d')[3])
	patches.append(pt.Patch(color=seaborn.color_palette('GnBu_d')[3], label='nfft=1024'))

	for i in range(9, 18):
		ax.get_children()[i].set_color(seaborn.color_palette('GnBu_d')[2])
	patches.append(pt.Patch(color=seaborn.color_palette('GnBu_d')[2], label='nfft=2048'))

	for i in range(18, 20):
		ax.get_children()[i].set_color(seaborn.color_palette('GnBu_d')[1])
	patches.append(pt.Patch(color=seaborn.color_palette('GnBu_d')[1], label='nfft=4096'))
	return patches


if __name__ == '__main__':
	hist_in = open('histories/training_history_10.pickle', 'rb')
	history = pickle.load(hist_in)
	hist_in.close()
	evalres_in = open('evalresults/evalres.pickle', 'rb')
	evalres = pickle.load(evalres_in)
	evalres_in.close()
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

	f3, ax = pl.subplots(figsize=(16, 9))
	ax.set_ylim([.9, 1])
	ax.set_xlabel('winlen')
	ax.set_ylabel('Precision Score')
	autolabel(ax.bar(evalres['winlen'], evalres['precision']))
	ax.legend(handles=patch_handle())

	f4, ax = pl.subplots(figsize=(16, 9))
	ax.set_ylim([.9, 1])
	ax.set_xlabel('winlen')
	ax.set_ylabel('ROC Score')
	autolabel(ax.bar(evalres['winlen'], evalres['roc_score']))
	ax.legend(handles=patch_handle())

	pl.show()
