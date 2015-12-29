package common.ml

import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.regression.LabeledPoint
import walmart.data.VectorCreation


class LogisticRegression(sc: SparkContext) {
  def fit = {


    val vectorCreation = new VectorCreation(sc)
    val labelPoints = vectorCreation.createVector

    val splitLabelPoints = labelPoints.randomSplit(Array(0.9, 0.1), 1212);

    //labelPoints.coalesce(1).saveAsTextFile("src/main/resources/vector")

    //val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    //val training = sqlContext.createDataFrame(splitLabelPoints(0)).toDF("label", "features")

    //    val lr = new LinearRegression()
    //    val paramGrid = new ParamGridBuilder()
    //      .addGrid(lr.regParam, Array(0.1, 0.01))
    //      .addGrid(lr.fitIntercept)
    //      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
    //      .build()
    //
    //    // In this case the estimator is simply the linear regression.
    //    // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    //    val trainValidationSplit = new TrainValidationSplit()
    //      .setEstimator(lr)
    //      .setEvaluator(new RegressionEvaluator())
    //      .setEstimatorParamMaps(paramGrid)
    //      // 80% of the data will be used for training and the remaining 20% for validation.
    //      .setTrainRatio(0.8)
    //
    //    // Run train validation split, and choose the best set of parameters.
    //    val model = trainValidationSplit.fit(training)
    //
    //    // Make predictions on test data. model is the model with combination of parameters
    //    // that performed best.
    //    val test = sqlContext.createDataFrame(splitLabelPoints(1)).toDF("label", "features")
    //    model.transform(test)
    //      .select("label", "prediction")
    //      .show()


    val modelMLIB = new LogisticRegressionWithLBFGS()
      .setNumClasses(46)
      .run(splitLabelPoints(0))

    val predictionAndLabels = splitLabelPoints(1).map { case LabeledPoint(label, features) =>
      val prediction = modelMLIB.predict(features)
      (prediction, label)
    }

//    val metrics = new MulticlassMetrics(predictionAndLabels)
//    val precision = metrics.precision
//    println("Precision = " + precision)
//
//    val MSE = predictionAndLabels.map { case (v, p) => math.pow((v - p), 2) }.mean()
//    println("training Mean Squared Error = " + MSE)
  }
}
