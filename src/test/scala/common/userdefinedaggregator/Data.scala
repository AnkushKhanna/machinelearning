package common.userdefinedaggregator

import org.apache.spark.SparkContext
import org.apache.spark.sql.{DataFrame, SQLContext}

object Data extends Serializable {
  def getSessionData(sc: SparkContext): DataFrame = {
    val sqlContext = new SQLContext(sc)
    sqlContext.createDataFrame(Seq(
      ("d1mm9tcy42", "lookup", "", "", "Windows Desktop", 319.0),
      ("d1mm9tcy42", "lookup", "", "", "Windows Desktop", 683.0),
      ("d1mm9tcy42", "search_results", "click", "view_search_results", "Windows Desktop", 59274.0),
      ("d1mm9tcy42", "lookup", "", "", "Windows Desktop", 95.0)
    )).toDF("user_id", "action", "action_type", "action_detail", "device_type", "secs_elapsed")
  }
}
