import numpy as np
import pandas as pd
from itertools import product, starmap, ifilter, groupby
from datetime import datetime
from operator import itemgetter
import time

np.seterr(divide='ignore', invalid='ignore') # supress divide by zero warning

def rmse(I,R,M,U,round=False):
	return np.sqrt(np.sum(np.square(np.multiply(I, (R-np.dot(U.T,M)))))/np.count_nonzero(I))

def levenshtein(source, target):
	if len(source) < len(target):
		return levenshtein(target, source)

	# So now we have len(source) >= len(target).
	if len(target) == 0:
		return len(source)

	# We call tuple() to force strings to be used as sequences
	# ('c', 'a', 't', 's') - numpy uses them as values by default.
	source = np.array(tuple(source))
	target = np.array(tuple(target))

	# We use a dynamic programming algorithm, but with the
	# added optimization that we only need the last two rows
	# of the matrix.
	previous_row = np.arange(target.size + 1)
	for s in source:
		# Insertion (target grows longer than source):
		current_row = previous_row + 1

		# Substitution or matching:
		# Target and source items are aligned, and either
		# are different (cost of 1), or are the same (cost of 0).
		current_row[1:] = np.minimum(current_row[1:], np.add(previous_row[:-1], target != s))

		# Deletion (target grows shorter than source):
		current_row[1:] = np.minimum(current_row[1:], current_row[0:-1] + 1)

		previous_row = current_row

	return previous_row[-1]

def resetRatings(x):
	x['rating'] = 0
	return x

# change the timestamp of unix to datetime type.
def convertTimestamp(x):
	x['timestamp'] = datetime.fromtimestamp(int(x['timestamp'])).strftime('%Y-%m-%d %H:%M:%S')
	return x

def getKey(item):
	return item['timestamp']

start_time = time.time()

rating_headers = ['userId', 'movieId', 'rating', 'timestamp']
ratings = pd.read_table('datasets/ml-latest-small/ratings.csv', sep=',', header=0, names=rating_headers)

trainingset_with_timestamp = ratings.sample(frac=0.7,random_state=200)
testset_with_timestamp = ratings.drop(trainingset_with_timestamp.index)


## get the sequences of all users in training set


sequences_training_userIds_movieIds = list()
for g, data in trainingset_with_timestamp.groupby('userId'):
	data.sort(['timestamp'], ascending=[True], inplace=True)
	sequences_training_userIds_movieIds.append((g, list(data['movieId'])))
	
casterian_sequences_training_userIds_movieIds = list(product(sequences_training_userIds_movieIds, sequences_training_userIds_movieIds))

levenshtein_casterian_sequences_training_userIds_movieIds = [(item[0][0], item[1][0], 1-(levenshtein(item[0][1], item[1][1])/float(max(len(item[0][1]), len(item[1][1]))))) for item in casterian_sequences_training_userIds_movieIds]
levenshtein_casterian_sequences_training_userIds_movieIds = list(ifilter(lambda x: x[2] > 0, levenshtein_casterian_sequences_training_userIds_movieIds))


similar_levenshtein_casterian_sequences_training_userIds_movieIds = [(k, [item[1] for item in g]) for k, g in groupby(levenshtein_casterian_sequences_training_userIds_movieIds, lambda x: x[0])]
#print (similar_levenshtein_casterian_sequences_training_userIds_movieIds)

# ALS Parameters
l = 0.1				# Regularization parameter lambda
K = 5				# convergence criterion
r = 12				# Dimensionality of latent feature space

error = []

print ('Start for training...')
#df.loc[~df.index.isin(t)]
idx = 0
for item in similar_levenshtein_casterian_sequences_training_userIds_movieIds:
	idx += 1
	print ('index = %s, userId = %s' % (idx, item[0]))
	parsed_testset_with_timestamp = testset_with_timestamp.loc[testset_with_timestamp['userId'] == item[0]]
	if parsed_testset_with_timestamp.empty:
		continue
	parsed_trainingset_with_timestamp = trainingset_with_timestamp.loc[trainingset_with_timestamp['userId'].isin(item[1])]
	if parsed_trainingset_with_timestamp.empty:
		continue
		
	reseted_ratings = pd.concat([parsed_trainingset_with_timestamp, parsed_testset_with_timestamp])
	
	reseted_ratings = reseted_ratings.apply(resetRatings, axis = 1)

	reseted_ratings = reseted_ratings.pivot_table(columns=['userId'],index=['movieId'],values='rating')
	reseted_ratings = reseted_ratings.fillna(0); # Replace NaN

	#trainingset_with_timestamp.pivot_table(columns=['userId'],index=['movieId'],values='rating')
	R = reseted_ratings.copy()
	R.update(parsed_trainingset_with_timestamp.pivot_table(columns=['userId'],index=['movieId'],values='rating'), join = 'left', overwrite = True)
	R = R.fillna(0); # Replace NaN
	R = R.as_matrix()

	Rtst = reseted_ratings.copy()
	Rtst.update(parsed_testset_with_timestamp.pivot_table(columns=['userId'],index=['movieId'],values='rating'), join = 'left', overwrite = True)
	Rtst = Rtst.fillna(0); # Replace NaN
	Rtst = Rtst.as_matrix()

	# Indicator Matrices
	I = np.copy(R)
	I[I > 0] = 1
	I[I == 0] = 0
	I2 = np.copy(Rtst)
	I2[I2 > 0] = 1
	I2[I2 == 0] = 0

	m,n = R.shape # Number of users and items
	k = 0	# convergence iterator
	
	#train_errors = []
	test_errors = []
	
	# Initialize matricies
	U = 3 * np.random.rand(r,m) # Latent user feature matrix (pattern matrix)
	M = 3 * np.random.rand(r,n) # Latent movie feature matrix (coefficient matrix)
	avg_movies = np.true_divide(R.sum(0),(R!=0).sum(0))
	avg_movies[np.isnan(avg_movies)] = 0
	M[0,:] = avg_movies # Set first row of Q to column vector of average ratings
	E = np.eye(r,dtype=int) # rxr idendity matrix
	
	print ('Start repeating...')
	# Repeat until convergence
	while k < K:
		# Fix M and solve for U
		for i, Ii in enumerate(I):
			nui = np.count_nonzero(Ii) # Number of items user i has rated
			if (nui == 0): nui = 1 # remove zeros
			
			# Least squares solution
			Ai = np.add(np.dot(M, np.dot(np.diag(Ii), M.T)), np.multiply(l, np.multiply(nui, E))) # A_i = M_{I_i}M_{I_i}^T + ln_{u_i}E
			Vi = np.dot(M, np.dot(np.diag(Ii), R[i].T)) # V_i = M_{I_i}R^T(i,I_i)
			U[:,i] = np.linalg.solve(Ai,Vi)

		# Fix U and solve for M
		for j, Ij in enumerate(I.T):
			nmj = np.count_nonzero(Ij) # Number of users that rated item j
			if (nmj == 0): nmj = 1 # remove zeros
			
			# Least squares solution
			Aj = np.add(np.dot(U, np.dot(np.diag(Ij), U.T)), np.multiply(l, np.multiply(nmj, E))) # A_j = U_{I_j}U_{I_j}^T + ln_{m_j}E
			Vj = np.dot(U, np.dot(np.diag(Ij), R[:,j])) # V_j = U_{I_j}R(I_j,j)
			M[:,j] = np.linalg.solve(Aj,Vj)

		#train_rmse = rmse(I,R,M,U)
		test_rmse = rmse(I2,Rtst,M,U)
		#train_errors.append(train_rmse)
		test_errors.append(test_rmse)
		print "[k: %d/%d] test-RMSE = %f" %(k+1, K, test_rmse)
		#print "[k: %d/%d] train-RMSE = %f  test-RMSE = %f" %(k+1, K, train_rmse, test_rmse)
		k = k + 1

	error.append(test_errors[K-1])

print ("For rank %d the RMSE is %f" %(r, np.mean(error)))
print("--- %s seconds ---" % (time.time() - start_time))
