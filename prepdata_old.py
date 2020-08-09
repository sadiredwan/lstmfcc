import pickle
import librosa
import numpy as np
import pandas as pd
from python_speech_features import mfcc


class Config:
	"""
	Mel Filterbanks = 26 * 99
	MFCC = 13 * 99

	Sampling Rate = 22050Hz
	NFFT = 551 samples
	"""
	def __init__(self, nfilt=26, nfeat=13, nfft=551):
		self.nfilt = nfilt
		self.nfeat = nfeat
		self.nfft = nfft


if __name__ == '__main__':
	config = Config()
	X = []
	y = []
	df = pd.read_csv('datamaps/datamap.csv')
	for i in range(len(df['fname'])):
		c = df.iloc[i]['class']
		f = df.iloc[i]['fname']
		signal, rate = librosa.load('data/'+c+'/'+f)
		signal = np.pad(signal, (0, rate-len(signal)), 'constant')
		signal = mfcc(signal, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
		X.append(signal)
		y.append(df.iloc[i]['label'])

	X, y = np.array(X), np.array(y)
	X_out = open('trainable/X.pickle', 'wb')
	pickle.dump(X, X_out)
	y_out = open('trainable/y.pickle', 'wb')
	pickle.dump(y, y_out)
	X_out.close()
	y_out.close()
	print('data has been written in /trainable')
