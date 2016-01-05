package airbnb

import airbnb.train.Train
import common.evaluator.MultiClassConfusionMetrics
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

object PredictStart {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .set("spark.driver.maxResultSize", "3g")
      .setAppName("airbnb")
      .setMaster("local[4]")
      .set("spark.driver.cores", "5")

    val runningFinal = false

    val sc = new SparkContext(conf)

    if (runningFinal) {}
    else {
      val sqlContext = new SQLContext(sc)
      val trainSessions = sqlContext.read.load("/Users/ankushkhanna/Documents/kaggle/airbnb/train_session")

      val finalValues =
        for {i <- 0.3 to 0.9 by 0.1
             Array(trainingData, testData, crossValidationData) = trainSessions.randomSplit(Array(i - 0.2, 1.0 - i, 0.2))

             model = new Train().train(trainingData)

             predictionsTraining = model.transform(trainingData).select("prediction", "indexed_label")
             f1Training = new MultiClassConfusionMetrics().predict(predictionsTraining)

             predictionsCV = model.transform(crossValidationData).select("prediction", "indexed_label")
             f1CV = new MultiClassConfusionMetrics().predict(predictionsCV)
        } yield (i, f1Training, f1CV)

      finalValues.foreach {
        case (i: Double, f1T: Double, f1CV: Double) => println(i + "  " + f1T + "  " + f1CV)
      }

    }
  }
}
