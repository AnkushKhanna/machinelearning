package airbnb.train

import common.ml.OneVsRestC
import org.apache.spark.sql.DataFrame

class Train {
  def train(train: DataFrame, regularization: Double, featureCol : String = "features") = {
    val algo = new OneVsRestC(regularization)
    algo.fit(train, "country_destination", featureCol, "indexed_label", "prediction_label")
  }
}
