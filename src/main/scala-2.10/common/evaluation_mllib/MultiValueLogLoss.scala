package common.evaluation_mllib

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD

class MultiValueLogLoss {

  def evaluate(data: RDD[LabeledPoint], model: RandomForestModel) = {
    val labelAndPreds = data.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
  }

}
