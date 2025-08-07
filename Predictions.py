import requests

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.sql.session import SparkSession

spark = SparkSession.builder.master('local').appName('PA2_Predictions').getOrCreate()

training_data = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/home/ubuntu/PA2/TrainingDataset.csv')
test_data = spark.read.format('csv').options(header='true', inferSchema='true', sep=';').load('/home/ubuntu/PA2/ValidationDataset.csv')

for data_frame in [training_data, test_data]:
    for col_header in data_frame.columns:
        data_frame = data_frame.withColumnRenamed(col_header, 'label' if 'quality' in col_header else  col_header.strip('"'))

assembler = VectorAssembler(
    inputCols=["fixed acidity",
               "volatile acidity",
               "citric acid",
               "residual sugar",
               "chlorides",
               "free sulfur dioxide",
               "total sulfur dioxide",
               "density",
               "pH",
               "sulphates",
               "alcohol"],
                outputCol="features_in")

scaler = StandardScaler(inputCol="features_in", outputCol="features", withStd=True, withMean=True)

log_regress = LogisticRegression()

regress_pipe = Pipeline(stages=[assembler, scaler, log_regress])

param_grid = ParamGridBuilder().build()

evaluator = MulticlassClassificationEvaluator(metricName="f1")

cv = CrossValidator(estimator=regress_pipe,  
                         estimatorParamMaps=param_grid,
                         evaluator=evaluator, 
                         numFolds=3
                        )

lr_cv_fit = cv.fit(training_data) 
print("F1 Score for Our Model: ", evaluator.evaluate(lr_cv_fit.transform(test_data)))
