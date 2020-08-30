import pickle
from keras import Sequential
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Flatten


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
		model.add(LSTM(64, return_sequences=True))
		model.add(LSTM(64, return_sequences=True))
		model.add(TimeDistributed(Dense(64, activation='relu')))
		model.add(TimeDistributed(Dense(32, activation='relu')))
		model.add(TimeDistributed(Dense(16, activation='relu')))
		model.add(TimeDistributed(Dense(8, activation='relu')))
		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))
		model.summary()
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', sensitivity, specificity])
		return model


def make_dataset():
	X_in, y_in = open('trainable/X_100.pickle', 'rb'), open('trainable/y_100.pickle', 'rb')
	X, y = pickle.load(X_in), pickle.load(y_in)
	X_in.close()
	y_in.close()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
	X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=2)
	X_test_out = open('testdata/X_test_100.pickle', 'wb')
	pickle.dump(X_test, X_test_out)
	y_test_out = open('testdata/y_test_100.pickle', 'wb')
	pickle.dump(y_test, y_test_out)
	X_test_out.close()
	y_test_out.close()
	return X_train, X_val, y_train, y_val


if __name__ == '__main__':
	X_train, X_val, y_train, y_val = make_dataset()
	model = RNN(input_shape=X_train.shape[1:]).run()
	hist = model.fit(X_train, y_train, epochs=100, batch_size=50, shuffle='true', validation_data=(X_val, y_val))
	hist_out = open('histories/training_history_100.pickle', 'wb')
	pickle.dump(hist.history, hist_out)
	hist_out.close()
	model.save('models/rnnmodel.h5')
