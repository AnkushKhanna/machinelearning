package airbnb

import airbnb.data.train.VectorCreationTrain
import org.apache.spark.{SparkConf, SparkContext}

object TrainStart {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .set("spark.driver.maxResultSize", "3g")
      .setAppName("airbnb")
      .setMaster("local[4]")
      .set("spark.driver.cores", "5")

    val runningFinal = false

    val sc = new SparkContext(conf)

    if (runningFinal)
    {}
    else {
     new VectorCreationTrain(sc).createVector(
        "/Users/ankushkhanna/Documents/kaggle/airbnb/train_users_2.csv",
        "/Users/ankushkhanna/Documents/kaggle/airbnb/vectorSession.csv/part-00000")
    }
  }
}
