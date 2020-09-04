import os
import re
import pickle
import numpy as np
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import roc_curve, auc, precision_score


def sensitivity(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	return true_positives / (possible_positives + K.epsilon())


def specificity(y_true, y_pred):
	true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
	possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
	return true_negatives / (possible_negatives + K.epsilon())


if __name__ == '__main__':
	list_winlen = ['5ms', '10ms', '15ms', '20ms', '25ms', '30ms', '35ms', '40ms', '45ms', '50ms',
	'55ms', '60ms', '65ms', '70ms', '75ms', '80ms', '85ms', '90ms', '95ms', '100ms']
	list_nfft = [128, 512, 512, 512, 1024, 1024, 1024, 1024, 1024,
	2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048, 4096, 4096]
	evalres = {'winlen':list_winlen, 'nfft':list_nfft, 'precision':[], 'roc_score':[]}
	PATH = os.getcwd()
	model_path = PATH + '/models'
	model_list = os.listdir(model_path)
	for model_name in model_list:
		n = re.findall('[0-9]+', model_name)
		model = load_model('models/'+model_name, custom_objects={'sensitivity': sensitivity, 'specificity': specificity}, compile=True)
		X_test_in, y_test_in = open('testdata/X_test_'+n[0]+'.pickle', 'rb'), open('testdata/y_test_'+n[0]+'.pickle', 'rb')
		X_test, y_test = pickle.load(X_test_in), pickle.load(y_test_in)
		X_test_in.close()
		y_test_in.close()
		y_pred = np.round(model.predict(X_test))
		evalres['precision'].append(precision_score(y_test, y_pred))
		fp, tp, thresholds = roc_curve(y_test, y_pred)
		evalres['roc_score'].append(auc(fp, tp))

	evalres_out = open('evalresults/evalres.pickle', 'wb')
	pickle.dump(evalres, evalres_out)
	evalres_out.close()
	print('results have been written in /evalresults')
