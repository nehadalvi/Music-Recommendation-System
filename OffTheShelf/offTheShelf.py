#Program for implementation of off-the-shelf MLLib ALS algorithm for recommendations 
import sys
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS

#Split the input on tab
def splitInput(inputLine):
  part = inputLine.split("\t")
  return int(part[0]), int(part[1]), float(part[2])


if len(sys.argv) < 4:
	print >> sys.stderr, "Usage: OffTheShelf<datafile>"
	exit(-1)

sc = SparkContext(appName="spark-mlib-als")

#To read the command line arguments
inputLines = sys.argv[1]
songNameswithId = sys.argv[2]
userId = int(sys.argv[3])

#Split input file to get userId with its item and rating
userItem = sc.textFile(inputLines).map(splitInput).cache()

#To get items that have been rated by the given user
userRatedItems = userItem.filter(lambda x: x[0] == userId).map(lambda x: x[1]).collect()

#To get item ids of all items
items = userItem.map(lambda x: x[1]).distinct()

#To get items that have not been rated by the given user
userUnrated = items.filter(lambda x: x not in userRatedItems).collect()

#Call to the train function from Spark MLlib ALS
model = ALS.train(userItem, rank=1, iterations=1)

#To get top 50 recommendations for user
topRecos = model.predictAll(sc.parallelize(map(lambda item: (userId, item), userUnrated))).map(
        lambda x: (x[1], x[2])).takeOrdered(50, key = lambda x: -x[1])

#Process the file containing song name and song ID to and generate a map of names with ids
songNames = sc.textFile(songNameswithId).map(lambda x: x.split("\t")).map(lambda x: (int(x[0]), x[1])).collectAsMap()
                 
                 
#Write the output to a file "OffTheShelfRecommendations"
with open("OffTheShelfRecommendations.txt", "w") as obj:
	obj.write("\n".join(map(lambda x: songNames.get(x[0], "N/A"), topRecos)))

sc.stop()