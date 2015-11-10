package walmart.evaluation

import common.evaluator.MultiClassLogLoss
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.DataFrame

class WalmartEvaluator {

  def evaluate(predictions: DataFrame) = {

    val evaluatorAcc = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("precision")


    val evaluatorLogLoss = new MultiClassLogLoss()
      .setLabelCol("indexedLabel")
      .setPredicationCol("prediction")
      .setProbabilityCol("probability")

    val logLoss = evaluatorLogLoss.evaluate(predictions)
    val accuracy = evaluatorAcc.evaluate(predictions)
    println("Log Loss = " + logLoss)
    println("Accuracy = " + accuracy)

    predictions.select("predictedLabel", "label", "features").show(5)

  }

}
