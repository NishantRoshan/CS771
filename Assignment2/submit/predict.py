import numpy as np
from numpy import random as rand
from catboost import CatBoostClassifier

# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# PLEASE BE CAREFUL THAT ERROR CLASS NUMBERS START FROM 1 AND NOT 0. THUS, THE FIFTY ERROR CLASSES ARE
# NUMBERED AS 1 2 ... 50 AND NOT THE USUAL 0 1 ... 49. PLEASE ALSO NOTE THAT ERROR CLASSES 33, 36, 38
# NEVER APPEAR IN THE TRAINING SET NOR WILL THEY EVER APPEAR IN THE SECRET TEST SET (THEY ARE TOO RARE)

# Input Convention
# X: n x d matrix in csr_matrix format containing d-dim (sparse) bag-of-words features for n test data points
# k: the number of compiler error class guesses to be returned for each test data point in ranked order

# Output Convention
# The method must return an n x k numpy nd-array (not numpy matrix or scipy matrix) of classes with the i-th row 
# containing k error classes which it thinks are most likely to be the correct error class for the i-th test point.
# Class numbers must be returned in ranked order i.e. the label yPred[i][0] must be the best guess for the error class
# for the i-th data point followed by yPred[i][1] and so on.

# CAUTION: Make sure that you return (yPred below) an n x k numpy nd-array and not a numpy/scipy/sparse matrix
# Thus, the returned matrix will always be a dense matrix. The evaluation code may misbehave and give unexpected
# results if an nd-array is not returned. Please be careful that classes are numbered from 1 to 50 and not 0 to 49.

map={1: 2.0, 19: 20.0, 3: 4.0, 0: 1.0, 8: 9.0, 9: 10.0, 2: 3.0, 37: 41.0, 27: 28.0, 14: 15.0, 7: 8.0, 12: 13.0, 39: 43.0, 10: 11.0, 35: 39.0, 33: 35.0, 44: 48.0, 28: 29.0, 15: 16.0, 6: 7.0, 22: 23.0, 4: 5.0, 38: 42.0, 31: 32.0, 23: 24.0, 24: 25.0, 18: 19.0, 25: 26.0, 20: 21.0, 29: 30.0, 32: 34.0, 41: 45.0, 36: 40.0, 46: 50.0, 11: 12.0, 5: 6.0, 16: 17.0, 43: 47.0, 34: 37.0, 40: 44.0, 17: 18.0, 42: 46.0, 13: 14.0, 45: 49.0, 30: 31.0, 21: 22.0, 26: 27.0}
def get_top_k(k,probs):
	global map
	probas = []
	for i in range(len(probs)):
		li = [(probs[i][j],j) for j in range(len(probs[i]))]
		li.sort(key = lambda a: a[0],reverse = True)
		probas.append([map[li[j][1]] for j in range(k)])
	return probas
    


def findErrorClass(X,k):
	# (X, y) = utils.loadData( "train", dictSize = dictSize )
	
	# Find out how many data points we have
	n = X.shape[0]

	
	X=X.toarray(order=None, out=None)
	
	
	# # Load and unpack a dummy model to see an example of how to make predictions
	# # The dummy model simply stores the error classes in decreasing order of their popularity
	# npzModel = np.load("catboost.sav")
	# # model = npzModel[npzModel.files[0]]
	# # # Let us predict a random subset of the 2k most popular labels no matter what the test point
	# yPred = npzModel.predict(X)
	# print(yPred)
	# shortList = model[0:2*k]
	# Make sure we are returning a numpy nd-array and not a numpy matrix or a scipy sparse matrix
	
	# for i in range( n ):
	# 	yPred[i,:] = rand.permutation( shortList )[0:k]

	
	cat1 = CatBoostClassifier()
	cat1.load_model('catboost.sav')
	y_Pred = cat1.predict(X)

	probs = cat1.predict_proba(X)
	train_pred = cat1.predict(X)
	y_Pred=get_top_k(k,probs)
	yPred=np.array(y_Pred)
	return yPred

