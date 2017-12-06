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

	small_ratings_data = small_ratings_raw_data.filter(lambda line: line!=small_ratings_raw_data_header).map(lambda line: line.split(",")).map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2])))
	training_rdd, test_rdd = small_ratings_data.randomSplit([7, 3], seed=0L)
	training_rdd = training_rdd.map(lambda x: (x['userId'], x['movieId'], x['rating']))
	test_rdd = test_rdd.map(lambda x: (x['userId'], x['movieId'], x['rating']))
	test_for_predict_rdd = test_rdd.map(lambda x: (x[0], x[1]))
	
	rank = 12					# rank is the number of latent factors in the model.
	seed = 5L							
	iterations = 5					# iterations is the number of iterations to run.
	regularization_parameter = 0.1			# lambda specifies the regularization parameter in ALS.
		
	model = ALS.train(training_rdd, rank, seed=seed, iterations=iterations, lambda_=regularization_parameter)
	## predict third element (rating) using the training model with ALS model.
	predictions = model.predictAll(test_for_predict_rdd).map(lambda r: ((r[0], r[1]), r[2]))
		
	## join these predictions with our test data (one contain the actual ratings, one contain predicted ratings).
	rates_and_preds = test_rdd.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
	## apply a squared difference and the we use the mean() action to get the MSE and apply sqrt.
	error_rates_and_preds = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

	print ('For rank %s the RMSE is %s' % (rank, error_rates_and_preds))
	sc.stop()
	print("--- %s seconds ---" % (time.time() - start_time))
	
############################################################################################################################################


#hoangducmanh@hoangducmanh:~/spark-2.0.2-bin-hadoop2.7/sbin$ ./start-master.sh 
#hoangducmanh@hoangducmanh:~/spark-2.0.2-bin-hadoop2.7/sbin$ ./start-slave.sh spark://hoangducmanh:7077
#hoangducmanh@hoangducmanh:~/spark-2.0.2-bin-hadoop2.7/bin$ ./spark-submit --master spark://hoangducmanh:7077 ~/Downloads/user-user/project.py
