package common.evaluator

import org.apache.spark.sql.DataFrame

class MisClassificationError {
  def predict(predictions: DataFrame, predictionCol: String, labelCol: String): Double = {
    val error = predictions.filter(predictions.col(predictionCol) !== predictions.col(labelCol))

    error.count().toDouble / predictions.count().toDouble
  }
}
