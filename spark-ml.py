from pyspark import SparkContext

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("PysparkExample")\
    .master("local[*]")\
    .config ("spark.sql.shuffle.partitions", "50")\
    .config("spark.driver.maxResultSize","5g")\
    .config ("spark.sql.execution.arrow.enabled", "true")\
    .getOrCreate()

dataframe  = spark.read.csv("data/Admission_Prediction.csv", header=True, inferSchema=True)
dataframe.show()

dataframe.describe().show()

type(dataframe)

dataframe.printSchema()

from pyspark.sql.functions import col
new_dataframe = dataframe.select(*(col(c).cast("float").alias(c) for c in dataframe.columns))
new_dataframe.printSchema()

from pyspark.sql.functions import col, count, isnan, when
for c in new_dataframe.columns:
  print(c)

from pyspark.ml.feature import Imputer
imputer = Imputer(inputCols=["GRE Score", "TOEFL Score","University Rating"],
                  outputCols=["GRE Score", "TOEFL Score","University Rating"])
model = imputer.fit(new_dataframe)

imputed_data = model.transform(new_dataframe)

imputed_data

#checking for null ir nan type values in our columns
imputed_data.select([count(when(col(c).isNull(), c)).alias(c) for c in imputed_data.columns]).show()

features = imputed_data.drop('Chance of Admit')
features

assembler = VectorAssembler( inputCols=features.columns,outputCol="features")
output = assembler.transform(imputed_data)

output= output.select("features", "Chance of Admit")
output = assembler.transform(imputed_data)
output= output.select("features", "Chance of Admit")
train_df,test_df = output.randomSplit([0.7, 0.3])
train_df.show()
test_df.show()

lin_reg = LinearRegression(featuresCol = 'features', labelCol='Chance of Admit')
linear_model = lin_reg.fit(train_df)
print("Coefficients: " + str(linear_model.coefficients))
print("Intercept: " + str(linear_model.intercept))

trainSummary = linear_model.summary
print("RMSE: %f" % trainSummary.rootMeanSquaredError)
print("r2: %f" % trainSummary.r2)

# prediction

predictions = linear_model.transform(test_df)
predictions.select("prediction","Chance of Admit","features").show()

from pyspark.ml.evaluation import RegressionEvaluator
pred_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="Chance of Admit",metricName="r2")
print("R Squared (R2) on test data =", pred_evaluator.evaluate(predictions))


featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(output)
featureIndexer = featureIndexer.transform(output)
new_indexed_data = featureIndexer.select("indexedFeatures", "Chance of Admit")
training, test = new_indexed_data.randomSplit([0.7, 0.3])
training.show()

test.show()

random_forest_reg = RandomForestRegressor(featuresCol="indexedFeatures",labelCol="Chance of Admit" )
# Train model.  This also runs the indexer.
model = random_forest_reg.fit(test)
# Make predictions.
predictions = model.transform(test)
predictions.show()

evaluator = RegressionEvaluator(labelCol="Chance of Admit", predictionCol="prediction", metricName="rmse")
print ("Root Mean Squared Error (RMSE) on test data = ",evaluator.evaluate(predictions))

evaluator = RegressionEvaluator(labelCol="Chance of Admit", predictionCol="prediction", metricName="r2")
print("R Squared (R2) on test data =", evaluator.evaluate(predictions))