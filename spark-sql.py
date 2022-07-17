import pandas as pd 
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import *
import time
#sc = SparkSession.builder.master("local[*]").getOrCreate()
sc = SparkSession.builder.appName("PysparkExample") \
    .config("spark.ui.port", "4050") \
    .getOrCreate()

df = sc.read.json("nyt2.json")
df.show(10)

df = df.dropDuplicates()
df.show(10)
df.select("author").show(10)
df.select("author","title", "rank", "price").show(10)
df.select("title",when(df.title!='ODD HOURS', 1).otherwise(0)).show(10)

df.filter((df.author).isin(['John Sandford'])).show(10)

df.select("author", "title", df.title.like('% THE %')).show(10)

df.select("author", "title", df.title.startswith("THE")).show(10)
df.select("author", "title", df.title.endswith("THE")).show(10)

df.select("author", "title", "rank", "description") \
    .write \
    .save("Rankings.parquet")





