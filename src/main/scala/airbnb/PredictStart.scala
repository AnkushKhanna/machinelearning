package airbnb

import airbnb.train.Train
import common.evaluator.{MisClassificationError, MultiClassConfusionMetrics}
import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

object PredictStart {

  def main(args: Array[String]) {
    val conf = new SparkConf()
      .set("spark.driver.maxResultSize", "3g")
      .setAppName("airbnb")
      .setMaster("local[4]")
      .set("spark.driver.memory", "8g")
      .set("spark.driver.cores", "5")

    val runningFinal = false

    val sc = new SparkContext(conf)

    if (runningFinal) {}
    else {
      val sqlContext = new SQLContext(sc)
      val trainSessions = sqlContext.read.load("/Users/ankushkhanna/Documents/kaggle/airbnb/train_session").cache()

      val Array(trainingData, testData, crossValidationData) = trainSessions.randomSplit(Array(0.6, 0.2, 0.2))
      //Polynomial expansion
      val finalValuePoly = calculatePloyExpansionTerm(trainingData, crossValidationData, 1)
      finalValuePoly.foreach {
        case (i: Int, errorTrain: Double, errorTest: Double) => println(i + "  " + errorTrain + "  " + errorTest)
      }
      //REGULARIZATION
      //
      //      val finalValuesRegularization = calculateRegularizationTerm(trainingData, testData, 0.0)
      //      finalValuesRegularization.foreach {
      //        case (i: Double, errorTrain: Double, errorTest: Double) => println(i + "  " + errorTrain + "  " + errorTest)
      //      }

      //LEARNING CURVE
      //      val finalValueLearningCurve = calculateLearningCurve(trainSessions, 0.0)
      //
      //      finalValueLearningCurve.foreach {
      //        case (i: Double, errorTrain: Double, errorCV: Double) => println(i + "  " + errorTrain + "  " + errorCV)
      //      }
    }

    def calculatePloyExpansionTerm(trainingData: DataFrame, crossValidationData: DataFrame, polyTerm: Int): List[(Int, Double, Double)] = {
      if (polyTerm == 3) {
        return Nil
      }
      val predictionCol = "prediction"
      val indexCol = "indexed_label"
      val (training, cv) =
        if (polyTerm == 1) {
          (trainingData, crossValidationData)
        } else {
          val polynomialExpansion = new PolynomialExpansion()
            .setInputCol("features")
            .setOutputCol("polyFeatures")
            .setDegree(polyTerm)
          val polyTraining = polynomialExpansion.transform(trainingData)
          val polyCV = polynomialExpansion.transform(crossValidationData)
          (polyTraining, polyCV)
        }
      val model =
        if (polyTerm == 1) {
          new Train().train(training, 0.0)
        } else {
          new Train().train(training, 0.0, "polyFeatures")
        }
      val predictionsTraining = model.transform(training).select(predictionCol, indexCol)
      val errorTraining = new MisClassificationError().predict(predictionsTraining, predictionCol, indexCol)

      val predictionsTest = model.transform(cv).select(predictionCol, indexCol)
      val errorCV = new MisClassificationError().predict(predictionsTest, predictionCol, indexCol)

      (polyTerm, errorTraining, errorCV) :: calculatePloyExpansionTerm(trainingData, crossValidationData, polyTerm + 1)
    }

    def calculateRegularizationTerm(training: DataFrame, test: DataFrame, regularization: Double): List[(Double, Double, Double)] = {
      if (regularization > 2) {
        return Nil
      }
      val predictionCol = "prediction"
      val indexCol = "indexed_label"

      val model = new Train().train(training, regularization)

      val predictionsTraining = model.transform(training).select(predictionCol, indexCol)
      val errorTraining = new MisClassificationError().predict(predictionsTraining, predictionCol, indexCol)

      val predictionsTest = model.transform(test).select(predictionCol, indexCol)
      val errorTest = new MisClassificationError().predict(predictionsTest, predictionCol, indexCol)

      val updatedReg =
        if (regularization == 0.0) {
          regularization + 0.01
        } else {
          regularization * 2
        }
      (regularization, errorTraining, errorTest) :: calculateRegularizationTerm(training, test, updatedReg)
    }
  }

  def calculateLearningCurve(trainSessions: DataFrame, regularization: Double): Seq[(Double, Double, Double)] = {
    val predictionCol = "prediction"
    val indexCol = "indexed_label"

    for {i <- 0.3 to 0.9 by 0.1
         Array(trainingData, testData, crossValidationData) = trainSessions.randomSplit(Array(i - 0.2, 1.0 - i, 0.2))

         model = new Train().train(trainingData, regularization)

         predictionsTraining = model.transform(trainingData).select(predictionCol, indexCol)
         errorTraining = new MisClassificationError().predict(predictionsTraining, predictionCol, indexCol)

         predictionsCV = model.transform(crossValidationData).select(predictionCol, indexCol)
         errorCV = new MisClassificationError().predict(predictionsCV, predictionCol, indexCol)
    } yield (i, errorTraining, errorCV)
  }
}
