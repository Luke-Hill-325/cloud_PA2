import sys
import random
import numpy as np
import pandas as pd
from functools import partial

from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col, desc

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder.appName("CS643_PA2").getOrCreate()

training_data = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/home/ubuntu/PA2/TrainingDataset.csv')
validation_data = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/home/ubuntu/PA2/ValidationDataset.csv')

for data_frame in [training_data, validation_data]:
    for col_header in data_frame.columns:
        data_frame = data_frame.withColumnRenamed(col_header, 'label' if 'quality' in col_header else  col_header.strip('"'))

vec_assembler = VectorAssembler(
    inputCols=[
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol"
    ],
    outputCol="features_in"
)


param_grid = ParamGridBuilder().build()

evaluator = MulticlassClassificationEvaluator(metricName="f1")

scaler = StandardScaler(inputCol="features_in", outputCol="features", withStd=True, withMean=True)

CommonCV = partial(
    CrossValidator,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3
)

common_stages = [vec_assembler, scaler]

log_regress = LogisticRegression()
lr_pipe = Pipeline(stages=common_stages + log_regress)
cv = CommonCV(estimator=lr_pipe)  

lr_cv_fit = cv.fit(training_data) 
print("F1 Score for Logistic Regression: ", evaluator.evaluate(lr_cv_fit.transform(validation_data)))

rand_forest = RandomForestClassifier()
rf_pipe = Pipeline(stages=common_stages + rand_forest)
cv = CommonCV(estimator=rf_pipe)

rf_cv_fit = cv.fit(training_data) 
print("F1 Score for Random Forest: ", evaluator.evaluate(rf_cv_fit.transform(validation_data)))

