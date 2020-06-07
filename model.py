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
	X_in, y_in = open('trainable/X.pickle', 'rb'), open('trainable/y.pickle', 'rb')
	X, y = pickle.load(X_in), pickle.load(y_in)
	X_in.close()
	y_in.close()
	return X, y


if __name__ == '__main__':
	X, y = make_dataset()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)
	model = RNN(input_shape=X.shape[1:]).run()
	hist = model.fit(X_train, y_train, epochs=15, batch_size=30, shuffle='true', validation_data=(X_test, y_test))
	hist_out = open('histories/training_history.pickle', 'wb')
	pickle.dump(hist.history, hist_out)
	hist_out.close()
