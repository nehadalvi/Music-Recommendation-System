We executed the algorithm for user id=7.

-----------------------------------------------------------------------------
>>Input:

Dataset has 2 files of the formats:

File1: song_names.txt
Fields: <songId> <songName>
Sample data:
1	The Great Kat
2	Lawrence The Kat
3	Kataklysm
4	Peter Kater
5	Kathy
6	Katia
7	Katie's Dimples

File2: ydata-ymusic-rating-study-v1_0-train.txt
Fields: <userId> <songId> <rating>
Sample data: 
1	14	5
1	35	1
1	46	1
1	83	1
1	93	1
1	94	1

-----------------------------------------------------------------------------
>>Command to execute Off the Shelf algorithm:

Argument[0]: offTheShelf.py
Argument[1]: Input dataset (File2)
Argument[2]: Input dataset (File1)
Argument[3]: User ID for which recommendation is to be given

spark-submit offTheShelf.py ydata-ymusic-rating-study-v1_0-train.txt song_names.txt 7

-----------------------------------------------------------------------------
>>Output:

Output is stored in file: OffTheShelfRecommendations.txt