package airbnb

import airbnb.data.session.VectorCreationSession
import org.apache.spark.{SparkConf, SparkContext}

object SessionStart {
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
      new VectorCreationSession(sc).createSessionVector(
        "/Users/ankushkhanna/Documents/kaggle/airbnb/",
        "sessions.csv",
        "vectorSession")
    }
  }
}
