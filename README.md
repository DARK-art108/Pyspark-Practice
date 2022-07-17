## PySpark-SQL and PySpark-ML(MLlib)

The repository contains python files for PySpark-SQL and PySpark-ML(MLlib).

In Pyspark-SQL you can find some SQL Queries which are performed on a sample data.In PySpark-ML you can a machine learning model which is trained on Admission_Prediction.csv data, two models are used linear regression and random forest. on a regression task.

Apache Spark uses MLlib to train a model.Spark-SQL is used to perform queries on the data.

Run the following command to run the code:

1. ```spark-submit --master local[*] --driver-memory 8g --executor-memory 8g --total-executor-cores 8 pyspark-sql.py```

2. ```spark-submit --master local[*] --driver-memory 8g --executor-memory 8g --total-executor-cores 8 pyspark-ml.py```

Note: Change the total-executor-cores to the number of cores you have.

The above commands will run the code on your local machine.