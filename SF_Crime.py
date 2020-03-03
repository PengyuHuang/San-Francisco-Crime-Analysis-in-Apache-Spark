# Databricks notebook source
# MAGIC %md
# MAGIC ## SF crime data analysis and modeling 

# COMMAND ----------

# DBTITLE 1,Import package 
from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
#from ggplot import *
import warnings

import os
os.environ["PYSPARK_PYTHON"] = "python3"


# COMMAND ----------

import urllib.request
urllib.request.urlretrieve("https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD", "/tmp/sf_03_18.csv")
dbutils.fs.mv("file:/tmp/sf_03_18.csv", "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv")
display(dbutils.fs.ls("dbfs:/laioffer/spark_hw1/data/"))

# COMMAND ----------

data_path = "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv"
# use this file name later

# COMMAND ----------

# DBTITLE 1,Data preprocessing
# read data from the data storage
# please upload your data into databricks community at first. 
crime_data_lines = sc.textFile(data_path)
#prepare data 
df_crimes = crime_data_lines.map(lambda line: [x.strip('"') for x in next(reader([line]))])
#get header
header = df_crimes.first()
print(header)

#remove the first line of data
crimes = df_crimes.filter(lambda x: x != header)

#get the first line of data
display(crimes.take(3))

#get the total number of data 
print(crimes.count())


# COMMAND ----------

# DBTITLE 1,Get dataframe and sql
from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df_opt1 = spark.read.format("csv").option("header", "true").load(data_path)
df_opt1.createOrReplaceTempView("sf_crime")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q1 question (OLAP): 
# MAGIC #####Write a Spark program that counts the number of crimes for different category.
# MAGIC 
# MAGIC Below are some example codes to demonstrate the way to use Spark RDD, DF, and SQL to work with big data. You can follow this example to finish other questions. 

# COMMAND ----------

# DBTITLE 1,Spark dataframe based solution for Q1
q1_result = df_opt1.groupBy('category').count().orderBy('count', ascending=False)
display(q1_result)

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Q1
#Spark SQL based
crimeCategory = spark.sql("SELECT  category, COUNT(*) AS Count FROM sf_crime GROUP BY category ORDER BY Count DESC")
display(crimeCategory)

# COMMAND ----------

# DBTITLE 1,Visualize your results
# important hints: 
## first step: spark df or sql to compute the statisitc result 
## second step: export your result to a pandas dataframe. 

crimes_pd_df = crimeCategory.toPandas()
#display(crimes_pd_df)
fig, axis = plt.subplots(1,1)
sb.barplot(x = "Count", y = "category", data = crimes_pd_df, color = "red")
plt.xlabel("Counts")
plt.tick_params(labelsize = 5)
plt.title("The number of crimes for different category", size = 15)
plt.grid(color = 'gray', linestyle='--', linewidth=0.5)
display(plt.show())
# Spark does not support this function, please refer https://matplotlib.org/ for visuliation. You need to use display to show the figure in the databricks community. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q2 question (OLAP)
# MAGIC Counts the number of crimes for different district, and visualize your results

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Q2
q2_result = spark.sql("select pdDistrict, count(*) as count from sf_crime group by pdDistrict order by count desc")
display(q2_result)

# COMMAND ----------

# DBTITLE 1,Visualize your results
q2_pd = q2_result.toPandas()
fig, axis = plt.subplots(1,1)
sb.barplot(x = "count", y = "pdDistrict", data = q2_pd, color = "red")
plt.tick_params(labelsize = 7)
#plt.grid(color = "gray", linestyle = "--", linewidth = "0.5")
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q3 question (OLAP)
# MAGIC Count the number of crimes each "Sunday" at "SF downtown".   
# MAGIC hints: SF downtown is defiend  via the range of spatial location. For example, you can use a rectangle to define the SF downtown, or you can define a cicle with center as well. Thus, you need to write your own UDF function to filter data which are located inside certain spatial range. You can follow the example here: https://changhsinlee.com/pyspark-udf/

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Q3
from pyspark.sql.types import IntegerType
def isSundaySF(dayOfWeek, X, Y):
  count = 0
  X = float(X)
  Y = float(Y)
  if dayOfWeek == "Sunday" and -122.4087 <= X <= -122.3933 and 37.7936 <= Y <= 37.7966:
    count = 1
  return count
crime_sunday_udf = udf(lambda x, y, z: isSundaySF(x, y, z), IntegerType())
df_sunday_count = df_opt1.select('DayOfWeek', 'date', 'X', 'Y', crime_sunday_udf('DayOfWeek', 'X', 'Y').alias("Sunday_Downtown"))
#display(df_sunday_count)
q3_result = df_sunday_count.filter(df_sunday_count.Sunday_Downtown == 1).groupBy(df_sunday_count.date).count().orderBy(df_sunday_count.date[6:9], df_sunday_count.date[0:2], df_sunday_count.date[3:5])
display(q3_result)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q4 question (OLAP)
# MAGIC Analysis the number of crime in each month of 2015, 2016, 2017, 2018. Then, give your insights for the output results. What is the business impact for your result?  

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Q4
q4_result = spark.sql("SELECT SUBSTRING(Date, 7, 4) AS year, SUBSTRING(Date, 0, 2) AS month, COUNT(*) AS count from sf_crime GROUP BY 1, 2 HAVING year IN(2015, 2016, 2017, 2018) ORDER BY 1, 2")
display(q4_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Insight of the result
# MAGIC We could see the crime is reduced a lot in 2018, especially in the winter of 2018, there was no crime.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q5 question (OLAP)
# MAGIC Analysis the number of crime the hour in certian day like 2015/12/15, 2016/12/15, 2017/12/15. Then, give your travel suggestion to visit SF. 

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Q5
q5_result = spark.sql("SELECT Date, SUBSTRING(Time, 0, 2) AS hour, COUNT(*) AS Count FROM sf_crime WHERE Date in('12/15/2015', '12/15/2016', '12/15/2017') GROUP BY Date, hour ORDER BY Date, hour")
display(q5_result)           

# COMMAND ----------

# DBTITLE 1,Spark Python based solution for Q5
from pyspark.sql.functions import *
getHour = udf(lambda x: int(x.split(':')[0]),  IntegerType())

df_opt2 = df_opt1.withColumn("Hour", getHour(col("Time")))
display(df_opt2)

# COMMAND ----------

q5_df = df_opt2.filter(df_opt2.Date.isin("12/15/2015", "12/15/2016","12/15/2017")).groupBy("Date", "Hour").count().orderBy("Date","Hour")
q5_2015 = q5_df.filter(q5_df.Date == "12/15/2015").toPandas()
q5_2016 = q5_df.filter(q5_df.Date == "12/15/2016").toPandas()
q5_2017 = q5_df.filter(q5_df.Date == "12/15/2017").toPandas()

fig, axis = plt.subplots(3,1)
sb.barplot(x = "Hour", y = "count", data = q5_2015, ax = axis[0]).set_title("2015")
sb.barplot(x = "Hour", y = "count", data = q5_2016, ax = axis[1]).set_title("2016")
sb.barplot(x = "Hour", y = "count", data = q5_2017, ax = axis[2]).set_title("2017")
plt.tight_layout()
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ####Insight from the result
# MAGIC 
# MAGIC The crime number increased at noon and after 16:00, so people should avoid hanging out during that time in SF.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Q6 question (OLAP)
# MAGIC (1) Step1: Find out the top-3 danger disrict  
# MAGIC (2) Step2: find out the crime event w.r.t category and time (hour) from the result of step 1  
# MAGIC (3) give your advice to distribute the police based on your analysis results. 

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Q6
q6_step1 = spark.sql("select PdDistrict, count(*) as count from sf_crime group by 1 order by count desc limit 3")
display(q6_step1)

# COMMAND ----------

q6_step1.createOrReplaceTempView("top3_danger_drt")

q6_result = spark.sql("SELECT PdDistrict, SUBSTRING(Time, 0, 2) AS hour, COUNT(*) as Count FROM sf_crime WHERE Category == 'WARRANTS' \
AND PdDistrict IN (SELECT PdDistrict FROM top3_danger_drt) GROUP BY 1, 2 order by 1, 2")
display(q6_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Insight from the result
# MAGIC There should be more police at daytime, especially from 12:00-17:00, and more police in Sounthern part.

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Q7 question (OLAP)
# MAGIC For different category of crime, find the percentage of resolution. Based on the output, give your hints to adjust the policy.

# COMMAND ----------

from pyspark.sql.types import StringType
res_per = spark.sql("SELECT res.Category AS Category, res.Resolution AS Resolution, tot.Total AS Total FROM \
                    ((SELECT Category, COUNT(*) As Resolution FROM sf_crime WHERE Resolution != 'NONE' GROUP BY Category)res \
                    LEFT OUTER JOIN \
                    (SELECT Category, COUNT(*) AS Total FROM sf_crime GROUP BY Category)tot \
                    ON res.Category = tot.Category)")
def percentage(resolution, total):
  resolution = float(resolution)
  total = float(total)
  percentage = round((resolution/total)*100, 0)
  return str(percentage) + "%"
percentage_udf = udf(lambda x, y: percentage(x, y), StringType())
q7_result = res_per.select("Category", "Resolution", "Total", percentage_udf("Resolution", "Total").alias("percentage")).orderBy("percentage", ascending = False)
display(q7_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ####Insight from result
# MAGIC Theft is the kind of crime with lowest resolution.

# COMMAND ----------

# MAGIC %md
# MAGIC ####Q8 question (Apply Spark ML clustering for spatial data analysis)
# MAGIC Extra: visualize the spatial distribution of crimes and run a kmeans clustering algorithm (please use Spark ML kmeans)

# COMMAND ----------

from pyspark.sql.types import DoubleType
def trans_double(x):
  return float(x)
trans_double_udf = udf(lambda x: trans_double(x), DoubleType())
kmeans_data = df_opt2.select('X', 'Y', trans_double_udf('X').alias('X_double'), trans_double_udf('Y').alias('Y_double'))
kmeans_data = kmeans_data.select('X_double', 'Y_double').where(kmeans_data.Y_double <= 40)
display(kmeans_data)

# COMMAND ----------

import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, PCA
import matplotlib.pyplot as plt
import warnings

cols = ['X_double', 'Y_double']
vectorAssembler = VectorAssembler(inputCols = cols, outputCol = 'features')
vdf = vectorAssembler.transform(kmeans_data)

kmeans = KMeans().setK(10).setSeed(1).setFeaturesCol("features") #18 is the counts of PdDistrict
model = kmeans.fit(vdf)

predictions = model.transform(vdf)
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)

print("Silhouette with squared euclidean distance = " + str(silhouette))

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
  print(center)

# COMMAND ----------

import warnings
pred = predictions.toPandas()
plt.style.use('ggplot')

fig, ax = plt.subplots()
ax.scatter(pred['X_double'], pred['Y_double'], c=(pred['prediction']),cmap=plt.cm.jet, alpha=0.9)
ax.set_title("the spatial distribution")
display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion. 
# MAGIC Everyone knows that the crime rate in San Fransisco is very high. I do this work to figure out the trend of crime rate and find methods to avoid that. Firstly, I upload the data by ETL process, and do some data clean to transform the data to the standard format. Than, I explored and processed the data to integrate data in an efficient format on Dataframe and Spark SQL for big data OLAP. 
# MAGIC I use pandas, matlabplot, and SparkSQL to analyze and visualize the data. Apply Spark ML clustering for spatial data analysisC to visualize the degree of safety distribution in SF.
# MAGIC From the result of analysis, we could see in SOUTHERN，MISIION，NORTHERN, crime rate is the highest, and during 12:00-17:00, crime rate is high. So people should avoid hanging out during that time and more police should be provided there, especially in the afternoon.
