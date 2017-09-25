File Name: user_based_rec.py 

I/O:
The program takes the following arguments:-
1. dataset => file containing data in the format: <user_id>  <song_id>  <rating>
   eg. file -> ydata-ymusic-rating-study-v1_0-train
   	1	14	5
	1	35	1
	1	46	1
	1	83	1
	1	93	1
	1	94	1
	1	153	5
2. songs_file => file containing data in the format: <song_id>  <song_name>
   eg. file -> song_names
   	1	The Great Kat
   	2	Lawrence The Kat
   	3	Kataklysm
   	4	Peter Kater
   	5	Kathy
   	6	Katia
   	7	Katie's Dimples
   	8	Boy Katindig
   	9	Katjuscha
  	10	Katmandu
3. number_of_recommendations => the desired number of recommendations <int> 
4. user_id => user_id of the user for whom recommendations are to be provided

O/P:
Generates a filename UserBasedRecUserId#.txt with the desired number of recommendations

Execution Command 
spark-submit user_based_rec.py <dataset> <songs_file> <number_of_recommendations> <user_id>
eg. spark-submit user_based_rec.py ydata-ymusic-rating-study-v1_0-train.txt song_names.txt 50 7