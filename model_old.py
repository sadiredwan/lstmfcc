import pickle
import numpy as np
import pandas as pd
from keras import Sequential
from scipy.io import wavfile
from keras import backend as K
from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Flatten


class Config:
	"""
	M(f) = 1125 ln(1 + f/700)
	M^-1(m) = 700(exp(m/1125) - 1)
	Mel Filterbanks = 26 * 99
	MFCC = 13 * 99

	Sampling Rate = 16kHz
	Window Length = 25 ms -> 400 samples
	Step Size = 10 ms
	NFFT = 512 samples (400 + 112 zero padded)
	"""
	def __init__(self, mode='rnn', nfilt=26, nfeat=13, nfft=512, rate=16000):
		self.mode = mode
		self.nfilt = nfilt
		self.nfeat = nfeat
		self.nfft = nfft
		self.rate = rate
		self.step = int(rate/10)


def sensitivity(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
	true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
	possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
	return true_negatives / (possible_negatives + K.epsilon())


class RNN:
	def __init__(self, input_shape):
		self.input_shape = input_shape
	
	def run(self):
		model = Sequential()
		model.add(LSTM(128, return_sequences=True, input_shape=self.input_shape))
		model.add(LSTM(128, return_sequences=True))
		model.add(Dropout(0.2)) 
		model.add(TimeDistributed(Dense(64, activation='relu')))
		model.add(Dropout(0.2))
		model.add(TimeDistributed(Dense(32, activation='relu')))
		model.add(Dropout(0.2))
		model.add(TimeDistributed(Dense(16, activation='relu')))
		model.add(Dropout(0.2))
		model.add(TimeDistributed(Dense(8, activation='relu')))
		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', sensitivity, specificity])
		return model


def make_dataset():
	X = []
	y = []
	df = pd.read_csv('datamaps/datamap.csv')
	df.set_index('fname', inplace=True)
	for f in df.index:
		rate, signal = wavfile.read('nfreduced/'+f)
		signal = mfcc(signal, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
		X.append(signal)
		y.append(df.loc[f]['label'])

	X, y = np.array(X), np.array(y)
	return X, y


def predict_class(fname):
	rate, signal = wavfile.read('nfreduced/'+fname)
	mfcc(signal, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
	signal = np.expand_dims(signal, axis=0)
	return model.predict(signal)


if __name__ == '__main__':
	config = Config(mode='rnn')
	X, y = make_dataset()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)
	model = RNN(input_shape=X.shape[1:]).run()
	hist = model.fit(X_train, y_train, epochs=10, batch_size=30, shuffle='true', validation_data=(X_test, y_test))
	hist_out = open('histories/training_history.pickle', 'wb')
	pickle.dump(hist.history, hist_out)
	hist_out.close()
