import pickle
import librosa
import numpy as np
import pandas as pd
from python_speech_features import mfcc


class Config:
	def __init__(self, samplerate, winlen, winstep, numcep, nfilt, nfft):
		self.samplerate = samplerate
		self.winlen = winlen
		self.winstep = winstep
		self.numcep = numcep
		self.nfilt = nfilt
		self.nfft = nfft
		self.lowfreq = 0
		self.highfreq = None
		self.preemph = 0.97
		self.ceplifter = 22
		self.appendEnergy = True


if __name__ == '__main__':
	config = Config(22050, 0.05, 0.01, 13, 26, 2048)
	X = []
	y = []
	df = pd.read_csv('datamaps/datamap.csv')
	for i in range(len(df['fname'])):
		c = df.iloc[i]['class']
		f = df.iloc[i]['fname']
		signal, rate = librosa.load('data/'+c+'/'+f)
		signal = np.pad(signal, (0, rate-len(signal)), 'constant')
		signal = mfcc(signal,
			samplerate=config.samplerate,
			winlen=config.winlen,
			winstep=config.winstep,
			numcep=config.numcep,
			nfilt=config.nfilt,
			nfft=config.nfft)
		X.append(signal)
		y.append(df.iloc[i]['label'])

	X, y = np.array(X), np.array(y)
	X_out = open('trainable/X_05.pickle', 'wb')
	pickle.dump(X, X_out)
	y_out = open('trainable/y_05.pickle', 'wb')
	pickle.dump(y, y_out)
	X_out.close()
	y_out.close()
	print('data has been written in /trainable')
