from __future__ import print_function

import sys
import os
import numpy as np
import pandas as pd
import datetime
import time

from pyspark import SparkContext
from pyspark.sql import SQLContext, Row

from pyspark.mllib.recommendation import ALS
import math
from pyspark.sql.types import *

############################################################################################################################################

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

############################################################################################################################################

def getKey(item):
	return item['timestamp']

############################################################################################################################################



if __name__ == "__main__":
	start_time = time.time()
	
	sc = SparkContext(appName = "BigDataAndDataMining")
	
	datasets_path = 'datasets'
	## set the path of the small dataset of ratings.csv.
	small_ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')

	## read the small dataset of ratings.csv.
	small_ratings_raw_data = sc.textFile(small_ratings_file)

	## filter out the header, included in each file.
	small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]

	small_ratings_data_with_timestamp = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header).map(lambda line: line.split(",")).map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2]), timestamp=int(p[3])))
	training_with_timestamp_rdd, test_with_timestamp_rdd = small_ratings_data_with_timestamp.randomSplit([7, 3], seed=0L)
	training_with_timestamp_rdd.values().cache()
	test_rdd = test_with_timestamp_rdd.map(lambda x: (x['userId'], x['movieId'], x['rating']))
	test_rdd.values().cache()
	
	#[(2, (150, 296, 590, 153, 349, 588, 339, 10, 161, 185, 208, 253, 593, 110, 50, 39, 364, 367, 356, 480, 261, 265, 350, 589, 168, 377, 500, 527, 223, 357, 272, 539, 551, 273, 587, 222, 248, 319, 485, 370, 371, 720, 52, 314, 372, 550, 661, 382))]
	key_training_with_timestamp_rdd = training_with_timestamp_rdd.keyBy(lambda x: x['userId'])
	groupby_key_training_with_timestamp_rdd = key_training_with_timestamp_rdd.groupByKey().map(lambda x : (x[0], sorted(list(x[1]), key=getKey)))
	sequence_groupby_key_training_with_timestamp_rdd = groupby_key_training_with_timestamp_rdd.map(lambda (x,y): (x, zip(*y)[0]))
	
	#[((2, (150, 296, 590, 153, 349, 588, 339, 10, 161, 185, 208, 253, 593, 110, 50, 39, 364, 367, 356, 480, 261, 265, 350, 589, 168, 377, 500, 527, 223, 357, 272, 539, 551, 273, 587, 222, 248, 319, 485, 370, 371, 720, 52, 314, 372, 550, 661, 382)), (2, (150, 296, 590, 153, 349, 588, 339, 10, 161, 185, 208, 253, 593, 110, 50, 39, 364, 367, 356, 480, 261, 265, 350, 589, 168, 377, 500, 527, 223, 357, 272, 539, 551, 273, 587, 222, 248, 319, 485, 370, 371, 720, 52, 314, 372, 550, 661, 382)))]
	cartesian_sequence_groupby_key_training_with_timestamp_rdd = sequence_groupby_key_training_with_timestamp_rdd.cartesian(sequence_groupby_key_training_with_timestamp_rdd)
	
	levenshtein_sequence_groupby_key_training_with_timestamp_rdd = cartesian_sequence_groupby_key_training_with_timestamp_rdd.map(lambda (x,y): (x[0], y[0], (1-levenshtein(x[1], y[1])/float(max(len(x[1]), len(y[1]))))))
	levenshtein_sequence_groupby_key_training_with_timestamp_rdd = levenshtein_sequence_groupby_key_training_with_timestamp_rdd.filter(lambda x: ((x[2] > 0))).map(lambda x: (x[0], x[1]))
	
	groupby_key_test_userId_training_userIds = levenshtein_sequence_groupby_key_training_with_timestamp_rdd.groupByKey().map(lambda x : (x[0], list(x[1])))
	
	error_single_rates_and_preds = []		# store errors of predicting on each user.
	
	rank = 12								# rank is the number of latent factors in the model.
	seed = 5L							
	iterations = 5							# iterations is the number of iterations to run.
	regularization_parameter = 0.1			# lambda specifies the regularization parameter in ALS.
	
	index = 0
	for test_userId_training_userIds in groupby_key_test_userId_training_userIds.collect():
		index += 1
		print ('index = %s, userId = %s' % (index, test_userId_training_userIds[0]))
		
		test_of_user_rdd = test_rdd.filter(lambda x: x[0] == test_userId_training_userIds[0])
		test_for_predict_rdd = test_of_user_rdd.map(lambda x: (x[0], x[1]))
		
		training_rdd = training_with_timestamp_rdd.filter(lambda x: x['userId'] in test_userId_training_userIds[1]).map(lambda x: (x['userId'], x['movieId'], x['rating']))
		
		model = ALS.train(training_rdd, rank, seed=seed, iterations=iterations, lambda_=regularization_parameter)
		## predict third element (rating) using the training model with ALS model.
		predictions = model.predictAll(test_for_predict_rdd).map(lambda r: ((r[0], r[1]), r[2]))
		
		## join these predictions with our test data (one contain the actual ratings, one contain predicted ratings).
		single_rates_and_preds = test_of_user_rdd.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
		## apply a squared difference and the we use the mean() action to get the MSE and apply sqrt.
		error_single_rates_and_preds.append(math.sqrt(single_rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean()))

	print ('For rank %s the RMSE is %s' % (rank, np.mean(error_single_rates_and_preds)))
	sc.stop()
	print("--- %s seconds ---" % (time.time() - start_time))
	
############################################################################################################################################


#hoangducmanh@hoangducmanh:~/spark-2.0.2-bin-hadoop2.7/sbin$ ./start-master.sh 
#hoangducmanh@hoangducmanh:~/spark-2.0.2-bin-hadoop2.7/sbin$ ./start-slave.sh spark://hoangducmanh:7077
#hoangducmanh@hoangducmanh:~/spark-2.0.2-bin-hadoop2.7/bin$ ./spark-submit --master spark://hoangducmanh:7077 ~/Downloads/user-user/project.py
