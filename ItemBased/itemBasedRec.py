#Program for item-based collaborative filtering using cosine similarity and weighted sums
import sys
from collections import defaultdict
from itertools import combinations
import numpy as np
import random
from pyspark import SparkConf,SparkContext


#To return user id and his associated items with their ratings. If a user exceeds 200 items, return any random 200 items out of those.
def randomItems(userId,itemAndRating):
    if len(itemAndRating) > 200:
        return userId, random.sample(itemAndRating,200)
    else:
        return userId, itemAndRating      


#Split the input line on tab into userId, itemId and rating and return
def splitInput(inputLine):
    inputLine = inputLine.split("\t")
    return inputLine[0],(inputLine[1],float(inputLine[2]))


#To find all (item,item) pairs for every user id and return its (id,id) and (rating,rating) pairs for that (item,item) pair
def findItemPairs(userId,itemAndRating):
    return [ [(item1[0],item2[0]), (item1[1],item2[1])] for (item1, item2) in combinations(itemAndRating,2)]

#To calculate and return cosine similarity and the no of raters for each item-item pair
def findCosSim(itemPair,itemsRatingPair):
    xx, xy, yy, cnt = (0.0, 0.0, 0.0, 0)
    for pairOfRating in itemsRatingPair:
        xx += np.float(pairOfRating[0]) * np.float(pairOfRating[0])
        yy += np.float(pairOfRating[1]) * np.float(pairOfRating[1])
        xy += np.float(pairOfRating[0]) * np.float(pairOfRating[1])
        cnt += 1
	#To calculate the cosine similarity
	numerator = xy
	denominator = np.sqrt(xx)*np.sqrt(yy)
    cosine_sim = (numerator / (float(denominator))) if denominator else 0.0
    return itemPair, (cosine_sim,cnt)

#To split the song id and song name on tab and return them as separate
def splitLine(line):
	line = line.split("\t")
	return line[0],line[1]


#To make the first item id of each item pair the key
def keyOnFirstItem(itemPair,itemSimilarity):
    (itemId1,itemId2) = itemPair
    return itemId1,(itemId2,itemSimilarity)


#Sorting the predictions list by similarity
def nearestNeighbor(itemId,itemSimilarity):
    
    itemSimilarity = list(itemSimilarity)
    itemSimilarity.sort(key=lambda x: x[1][0],reverse=True)
    return itemId, itemSimilarity


# To calculate top N recommendations using weighted sum
def topRecos(userId,itemsAndRating,itemSimMatrix,n):

    # Store all item scores in a dictionary
    itemScore = defaultdict(int)
    similaritySum = defaultdict(int)
	#To find the nearest neighbor for every item
    for (item,rating) in itemsAndRating:
        nearestNeighbors = itemSimMatrix.get(item,None)
        if nearestNeighbors:
            for (neighbor,(similarity,count)) in nearestNeighbors:
                if neighbor != item:
                    itemScore[neighbor] += similarity * rating
                    similaritySum[neighbor] += similarity
    # create the normalized list of scored items
    scoredItems = [(score/similaritySum[item],item) for item,score in itemScore.items()]
    scoredItems.sort(reverse=True)
    return userId,scoredItems[:n]



if len(sys.argv) < 4:
	print >> sys.stderr, "Usage: abc<datafile>"
	exit(-1)

conf = SparkConf()
sc = SparkContext(conf = conf)

#Taking the command line arguments
inputLines = sc.textFile(sys.argv[1])
songNamesWithID = sys.argv[2]
userId = sys.argv[3]
simDictionary = {}


#Split the input and group by user ID to get (userId,[(itemId1,itemRating1), (itemId2, itemRating2),...]) pair
inputSplit= inputLines.map(splitInput).groupByKey() 
userItem=inputSplit.map(lambda x: randomItems(x[0],x[1])).cache()


#Get all the item,item pairs and their ratings and group by (item,item) pairs.
#Output is in the form ((itemId1,itemId2),[(itemRating1,itemRating2)(itemRating1,itemRating2),....])
allItemPairs=userItem.filter(lambda x: len(x[1]) > 1).map(lambda x: findItemPairs(x[0],x[1]))
itemPairs=allItemPairs.flatMap(lambda x:x).groupByKey()


#To calculate cosine similarity, change the key to Id of item1, and calculate the nearest neighbors
#Output is of the form ((itemId1,itemId2),(similarity,noOfRaters))
itemSimMatrix = itemPairs.map(lambda x: findCosSim(x[0],x[1])).map(
lambda x: keyOnFirstItem(x[0],x[1])).groupByKey().map(
lambda x: nearestNeighbor(x[0],x[1])).collect()


#Store the item similarity matrix in a broadcast varialbe to access all over the cluster

for (item,data) in itemSimMatrix:
    simDictionary[item] = data
    simBrdCst = sc.broadcast(simDictionary)


#To find top 50 item recommendations for each user

userItemList = userItem.map(lambda x: topRecos(x[0],x[1],simBrdCst.value,50))

#Display top 50 recommendations for a given user

userRecs = sc.parallelize(userItemList.filter(lambda x:x[0]==userId).values().collect()[0])
songIds = userRecs.map(lambda x:x[1]).collect()

songNames = sc.textFile(songNamesWithID)
namesToIDMap = songNames.map(lambda x:x.encode("ascii","ignore")).map(splitLine).filter(lambda x:x[0] in songIds).map(lambda x:x[1]).collect()

#Store final 50 recommendations in "itemBasedRecommendation" file

#print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
#print "final"

with open("itemBasedRecommendation.txt", "w") as obj:
	obj.write("\n".join([str(x) for x in namesToIDMap]))

sc.stop()