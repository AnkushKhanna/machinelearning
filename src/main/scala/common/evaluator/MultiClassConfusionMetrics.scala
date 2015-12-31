package common.evaluator

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, Row}

class MultiClassConfusionMetrics {
  def predict(predictions: DataFrame): Double = {
    val predictionsAndLabels = predictions.map {case Row(p: Double, l: Double) => (p, l)}

    // compute confusion matrix
    val metrics = new MulticlassMetrics(predictionsAndLabels)
    metrics.fMeasure
  }
}
