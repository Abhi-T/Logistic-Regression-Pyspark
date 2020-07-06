#The intent of this code is to demostrate binary classification in pyspark

#Initializinf the spark session
from pyspark.sql import SparkSession
spark=SparkSession.builder.master('local').appName("diabeties").getOrCreate()

#reading the csv
raw_data=spark.read.format("csv").option("header","true").option("inferSchema", "true").load(r"diabeties.csv")
# print(raw_data.columns)

# print(raw_data.describe().select("Summary", "Pregnancies", "Glucose", "BloodPressure").show())
# print(raw_data.describe().select("Summary", "BMI", "DiabetesPedigreeFunction", "Age").show())

import numpy as np
from pyspark.sql.functions import when
raw_data=raw_data.withColumn("Glucose", when(raw_data.Glucose==0, np.nan).otherwise(raw_data.Glucose))
raw_data=raw_data.withColumn("BloodPressure",when(raw_data.BloodPressure==0,np.nan).otherwise(raw_data.BloodPressure))
raw_data=raw_data.withColumn("SkinThickness",when(raw_data.SkinThickness==0,np.nan).otherwise(raw_data.SkinThickness))
raw_data=raw_data.withColumn("BMI",when(raw_data.BMI==0,np.nan).otherwise(raw_data.BMI))
raw_data=raw_data.withColumn("Insulin",when(raw_data.Insulin==0,np.nan).otherwise(raw_data.Insulin))

# print(raw_data.select("Insulin", "Glucose", "BloodPressure", "SkinThickness", "BMI").show(5))

# So we have replaced all "0" with NaN. Now, we can simply impute the NaN by calling an imputer :)
from pyspark.ml.feature import Imputer
imputer=Imputer(inputCols=["Glucose","BloodPressure","SkinThickness","BMI","Insulin"],outputCols=["Glucose","BloodPressure","SkinThickness","BMI","Insulin"])
model=imputer.fit(raw_data)
raw_data=model.transform(raw_data)
# print(raw_data.show(5))

cols=raw_data.columns
cols.remove("Outcome")

#let us import the vector assembler
from pyspark.ml.feature import VectorAssembler
assembler=VectorAssembler(inputCols=cols,outputCol="features")

#now let us use the transform method
raw_data=assembler.transform(raw_data)
# print(raw_data.select("features").show(truncate=False))

#standard scaler
from pyspark.ml.feature import StandardScaler
standardscaler=StandardScaler().setInputCol("features").setOutputCol("scaled_features")
raw_data=standardscaler.fit(raw_data).transform(raw_data)
# print(raw_data.select("features", "scaled_features").show())

#train test split
train, test=raw_data.randomSplit([0.8, 0.2], seed=12345)

#let us check whether their is imbalance in the dataset
dataset_size=  float(train.select("Outcome").count())
numPositives=train.select("Outcome").where('Outcome == 1').count()
per_ones=(float(numPositives)/float(dataset_size))*100
numNegatives=float(dataset_size-numPositives)
# print('The number of ones are {}'.format(numPositives))
# print('Percentage of ones are {}'.format(per_ones))


#Imbalance Dataset
# In our dataset (train) we have 34.27 % positives and 65.73 % negatives. Since negatives are in a majority. Therefore,logistic loss objective function should treat the positive class (Outcome == 1) with higher weight. For this purpose we calculate the BalancingRatio as follows:
#
# BalancingRatio= numNegatives/dataset_size
#
# Then against every Outcome == 1, we put BalancingRatio in column "classWeights", and against every Outcome == 0, we put 1-BalancingRatio in column "classWeights"
#
# In this way, we assign higher weightage to the minority class (i.e. positive class)

BalancingRatio= numNegatives/dataset_size
# print(print('BalancingRatio = {}'.format(BalancingRatio)))

train=train.withColumn("classWeights", when(train.Outcome == 1,BalancingRatio).otherwise(1-BalancingRatio))
# print(train.select("classWeights").show(5))

#Feature Selection
# We use the ChiSqSelector provided by Spark ML for selecting significant features.
from pyspark.ml.feature import ChiSqSelector
css = ChiSqSelector(featuresCol='scaled_features',outputCol='Aspect',labelCol='Outcome',fpr=0.05)
train=css.fit(train).transform(train)
test=css.fit(test).transform(test)
print(test.select("Aspect").show(5, truncate=False))

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(labelCol="Outcome", featuresCol="Aspect",weightCol="classWeights",maxIter=10)
model=lr.fit(train)
predict_train=model.transform(train)
predict_test=model.transform(test)
predict_test.select("Outcome","prediction").show(10)

# Evaluating the model
# Now let us evaluate the model using BinaryClassificationEvaluator class in Spark ML.
# BinaryClassificationEvaluator by default uses areaUnderROC as the performance metric

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator=BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol="Outcome")

print(predict_test.select("Outcome", "rawPrediction", "prediction", "probability").show(5))
print("The area under ROC for train set is {}".format(evaluator.evaluate(predict_train)))
print("The area under ROC for test set is {}".format(evaluator.evaluate(predict_test)))

# Hyper parameters
# # To this point we have developed a classification model using logistic regression.
# However, the working of logistic regression depends upon the on a number of parameters. As of now we have worked with only the default parameters.
# # Now, let s try to tune the hyperparameters and see whether it make any difference.
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = ParamGridBuilder()\
    .addGrid(lr.aggregationDepth,[2,5,10])\
    .addGrid(lr.elasticNetParam,[0.0, 0.5, 1.0])\
    .addGrid(lr.fitIntercept,[False, True])\
    .addGrid(lr.maxIter,[10, 100, 1000])\
    .addGrid(lr.regParam,[0.01, 0.5, 2.0]) \
    .build()

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations
cvModel = cv.fit(train)
# this will likely take a fair amount of time because of the amount of models that we're creating and testing
predict_train=cvModel.transform(train)
predict_test=cvModel.transform(test)
print("The area under ROC for train set after CV  is {}".format(evaluator.evaluate(predict_train)))
print("The area under ROC for test set after CV  is {}".format(evaluator.evaluate(predict_test)))
