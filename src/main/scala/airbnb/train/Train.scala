package airbnb.train

import common.ml.OneVsRestC
import org.apache.spark.sql.DataFrame

class Train {
  def train(train: DataFrame) = {
    val algo = new OneVsRestC
    algo.fit(train, "country_destination", "features", "indexed_label", "prediction_label")
  }
}
