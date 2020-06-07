"""noise floor reduction"""

import os
import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile


def envelope(y, rate, threshold):
	mask = []
	y = pd.Series(y).apply(np.abs)
	y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
	for mean in y_mean:
		if mean > threshold:
			mask.append(True)
		else:
			mask.append(False)
	return mask


if __name__ == '__main__':
	df = pd.read_csv('datamaps/datamap.csv')

	if len(os.listdir('nfreduced')) == 0:
		for i in range(len(df['fname'])):
			c = df.iloc[i]['class']
			f = df.iloc[i]['fname']
			# original sr=22050, downsampled to 16000
			signal, rate = librosa.load('data/'+c+'/'+f, sr=16000)
			mask = envelope(signal, rate, 0.0005)
			signal = signal[mask]
			signal = np.pad(signal, (0, 16000-len(signal)), 'constant')
			wavfile.write(filename='nfreduced/'+f, rate=rate, data=signal)
