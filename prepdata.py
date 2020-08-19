import pickle
import librosa
import numpy as np
import pandas as pd


class Config:
	def __init__(self, n_fft, hop_length, n_mfcc):
		self.n_fft = n_fft
		self.hop_length = hop_length
		self.n_mfcc = n_mfcc


if __name__ == '__main__':
	config = Config(1024, 100, 13)
	X = []
	y = []
	df = pd.read_csv('datamaps/datamap.csv')
	for i in range(len(df['fname'])):
		c = df.iloc[i]['class']
		f = df.iloc[i]['fname']
		signal, rate = librosa.load('data/'+c+'/'+f)
		signal = np.pad(signal, (0, rate-len(signal)), 'constant')
		signal = librosa.feature.mfcc(signal, n_fft=config.n_fft, hop_length=config.hop_length, n_mfcc=config.n_mfcc)
		X.append(signal)
		y.append(df.iloc[i]['label'])

	X, y = np.array(X), np.array(y)
	X_out = open('trainable/X_100.pickle', 'wb')
	pickle.dump(X, X_out)
	y_out = open('trainable/y_100.pickle', 'wb')
	pickle.dump(y, y_out)
	X_out.close()
	y_out.close()
	print('data has been written in /trainable')
