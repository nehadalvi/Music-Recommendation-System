import sys
import numpy as np
from pyspark import SparkContext
from math import sqrt



# This function returns popular users sorted in descending orders of their similarity measures obtained using cosine similarity with the desired user
def computeSimilarity(songIds_ratings, power_users):
	return sorted(map(lambda power_user: [power_user[0], cosine_similarity(songIds_ratings, power_user[1])], power_users), key = lambda r: -r[1])
    

	
# This function computes cosine similarity	
def cosine_similarity(idRatings_1, idRatings_2):
	# idRatings_1 = [(song_id_1<int>, rating_1<float>)]
	# idRatings_2 = [(song_id_1<int>, rating_1<float>)]
	idRatings1 = sorted(idRatings_1, key = lambda x: x[0])  #sort by song_ids
	idRatings2 = sorted(idRatings_2, key = lambda x: x[0])  #sort by song_ids
	l1 = len(idRatings1)
	l2 = len(idRatings2)
	i, j = 0, 0   #initializing step => i and j are iterators
	similarity_measure = 0.0   # initial similarity_measure is 0.0
	
	while i < l1 and j < l2:
		if idRatings1[i][0] == idRatings2[j][0]: # found matching song_id
			similarity_measure += idRatings1[i][1] * idRatings2[j][1]
			i += 1
			j += 1
		elif idRatings1[i][0] < idRatings2[j][0]:
			i += 1
		else:
			j += 1	
	return similarity_measure

	
	
# This function predicts the ratings for the songs not heard/rated by the given user	
def predict_user_rating(songId, power_users, similarity_dict):
	predicted_rating = 0   # initial predicted rating is 0
	no_of_matchedIds = 0   # initially 0 songIds match 
	
	# for every <power_user and his/her list of ratings> in power_users
	for (user, song_ratings) in power_users:
		# for every <song_id, rating> in the list of a power_user's song_ratings
		for (song_id, rating) in song_ratings:
		    # check if the song_id matches the unrated songId 
			if song_id == songId:
				predicted_rating += similarity_dict[user] * rating   # if matches, predicted_rating += similarity_measure of the power_user * rating given by the power_user
				no_of_matchedIds += 1    # increment matched Ids by 1
				
	return predicted_rating / max(no_of_matchedIds, 1) # take max between 0 and 1 to avoid 0 / 0 for unmatched Ids

	
	
#The main program begins here
# input => spark-submit user_based.py <dataset> <songs_file> <number_of_recommendations> <user_id>	
if __name__ == "__main__":
  if len(sys.argv) !=5:  #check if the number of arguments entered are 6
    print >> sys.stderr, "Incorrect no of arguments"
    exit(-1)

  sc = SparkContext(appName="User Based Music Recommendation System")
  
  # the dataset
  input_file = sc.textFile(sys.argv[1])
  
  # song names file
  song_names_file = sc.textFile(sys.argv[2])
  
  # no of recommendations
  no_of_recom = int(sys.argv[3])
  
  # user-id of the user for whom recommendations is to be provided
  user_id = int(sys.argv[4])
  
  # no of popular users
  top_users = 1000  # for linear scalability
  
  '''
  data-set preprocessing
  # input_file line-> user_id  song_id  rating
  # split on tab
  # parse each row in the form -> int(user_id), (int(song_id),float(rating))
  '''
  input_file = input_file.map(lambda line: line.split("\t"))\
				.map(lambda row: (int(row[0]), (int(row[1]), float(row[2]))))
  
  '''
  song names file preprocessing
  # file -> <song_id>  <song_name>
  # split on tab, and map as -> [int(song_id), song_name]
  # collectAsMap
  '''
  song_names = song_names_file.map(lambda line: line.split("\t"))\
                  .map(lambda row: (int(row[0]),row[1]))
  song_names_map = song_names.collectAsMap()
  
  '''
  creating users_rdd
  # group by user ids => [(user_id_1,<iterable list>),(user_id_2,<iterable list>),...]
  # calculate sum of squares of ratings of all the songs rated by a user and take the sqrt => [(user_id_1, <iterable list(song id,rating)>, computed sqrt),...]
  # compute new rating for every song rated by a user by dividing the rating by the computed sqrt. To avoid divide by 0 instance, add 0.000001 to the denominator
  users_rdd => [(user_id_1, [(song_id_1, new_rating),(song_id_2, new_rating),...]),(...)]
  '''
  users_rdd = input_file.groupByKey()\
         .map(lambda row: (row[0], row[1], sqrt(sum(map(lambda rating: rating[1] * rating[1], row[1])))))\
         .map(lambda row: (row[0], map(lambda rating: (rating[0], rating[1] / (0.000001 + row[2])), row[1])))
				
  # get all the song id's
  song_ids = input_file.map(lambda item: item[1][0]).distinct()
  
  # fetch top 1000 power users who have rated maximum songs
  power_users = users_rdd.takeOrdered(top_users, key = lambda u: -len(u[1]))

  # fetch the desired user from all users_rdd => [(user_id, [(song_id, rating), (song_id, rating), ...])]
  user = users_rdd.filter(lambda u: u[0] == user_id).first()
  
  # fetch the id's of the songs rated by the user
  user_rated_songs = [song_rating[0] for song_rating in user[1]]
  
  # dictionary of similarity measures => {power_user_1: similarity_measure, power_user_2: similarity_measure, ...}
  similarity_dict = dict(computeSimilarity(user[1], power_users))
  
  # user unrated songs
  user_unrated_songs = song_ids.filter(lambda song: song not in user_rated_songs)
  
  # generate recommendations
  recommendations = user_unrated_songs.map(lambda songId: (songId, predict_user_rating(songId, power_users, similarity_dict)))\
                  .takeOrdered(no_of_recom, lambda r: -r[1])
  
  sc.stop()
  
  # write the recommendations to a file
  with open("UserBasedRecUserId%d.txt" % user_id, "w") as o:
    o.write("\n".join(map(lambda reco: song_names_map.get(reco[0], "title N/A"), recommendations)))
  
  
 
  