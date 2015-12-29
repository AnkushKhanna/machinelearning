package airbnb.data.train

import common.operations.{Write, Read}
import org.apache.spark.SparkContext

class VectorCreationTrain(sc: SparkContext) {
  def createVector(trainPath: String, sessionPath: String) = {
    val train = Read.csv(trainPath, sc)
    val sessions = Read.csv(sessionPath, sc)

    val train_session = train.join(sessions, train("id") === sessions("user_id"), joinType = "left_outer")

    Write.csv("/Users/ankushkhanna/Documents/kaggle/airbnb/train_session", train_session.select("id", "session-features"))
  }
}
