import os
import findspark

# set java and spark home directory as env, it can be done using spark-env or bash_profile instead...
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-2.3.4-bin-hadoop2.7/"

findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct
from pyspark.ml.clustering import KMeans
# from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler

# initialize SparkSession
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("CityBike_Spark") \
    .getOrCreate()

# set the log level to WARN
spark.sparkContext.setLogLevel('WARN')

# path to data
path_to_file = "Brisbane_CityBikeNew.json"

# read data 
df = spark.read.json(path_to_file, multiLine=True).cache()

# split data to create model only on few dat
df_train,df_target = df.randomSplit([0.4, 0.6])

# discover data by checking schema 
df_train.printSchema()
# df.describe().show()

# look at few rows
# df_train.show(5,False)

# create feature vector
vecAssembler = VectorAssembler(inputCols=["latitude", "longitude"], outputCol="features")

# add the feature vector to dataframe
new_df = vecAssembler.transform(df_train)

# Initialize a k-means model with k = 2, k need to be tunned later to find the best number of clusters according to lowest wsse.
kmeans = KMeans().setK(2).setSeed(1)

# train lat, long model
model = kmeans.fit(new_df.select('features'))

# compute global error which is the sum of each error per row
min_wsse = model.computeCost(new_df)

# print(f"model lat,long compute error : {min_wsse}")

for k in range(3,100_000): # 100_000 is arbitrary since having 100_000 is a lot of clusters and we won't reach it
  
  # Initialize a new k-means model with k cluster
  kmeans = KMeans().setK(k).setSeed(1)

  # train lat, long model with the new K
  model = kmeans.fit(new_df.select('features'))

  # compute the global error for this K
  wsse = model.computeCost(new_df)

  # compare if the min_wsse is higher than the newer wsse, if yes then the minimum = wsse else break cause the error will start rising again
  if min_wsse > wsse:
    min_wsse = wsse
  else:
    break # end the loop when newer wsse > minimum
  
  # print(f"model lat,long compute error : {min_wsse}")

# add the feature vector to dataframe
target_df = vecAssembler.transform(df)

# make predictions to dataframe and show it
transformed = model.transform(target_df)
wsse = model.computeCost(target_df)

print(f"model lat,long on whole dataframe compute error : {wsse} with {k} cluster")

# save results as parquet for later use if needed, why parquet ? to keep the schema 
transformed.write.mode("overwrite").parquet("lat_long_result.parquet")







