package airbnb.data.train

import airbnb.Helper
import common.operations.{Clean, Read, Write}
import common.transfomration.{THashing, TTokenize, Transform}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{Normalizer, VectorAssembler}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.{SaveMode, DataFrame, SQLContext, functions}

class VectorCreationTrain(sc: SparkContext) {
  def createVector(trainPath: String, sessionPath: String) = {
    val train = Read.csv(trainPath, sc)

    //val trainNoNDF = train.filter((train("country_destination") !== "NDF") || (train("country_destination") !== "US"))

    //Cleaning
    val train1 = train.withColumn("language_sanitized", Clean.remove("-unknown-", "")(train.col("language"))).drop(train.col("language"))

    //Transformation
    val transform = new Transform with TTokenize with THashing

    val train3 = getTransformedDataFrame(train1, transform, "signup_method", "signup_method-features", 4).drop(train1.col("signup_method"))
    //val train3 = getTransformedDataFrame(train2, transform, "signup_flow", "signup_flow-features", 18).drop(train.col("signup_flow"))
    val train4 = getTransformedDataFrame(train3, transform, "language_sanitized", "language_sanitized-features", 26).drop(train3.col("language_sanitized"))
    val train5 = getTransformedDataFrame(train4, transform, "affiliate_channel", "affiliate_channel-features", 8).drop(train4.col("affiliate_channel"))
    val train6 = getTransformedDataFrame(train5, transform, "affiliate_provider", "affiliate_provider-features", 18).drop(train5.col("affiliate_provider"))
    val train7 = getTransformedDataFrame(train6, transform, "first_affiliate_tracked", "first_affiliate_tracked-features", 8).drop(train6.col("first_affiliate_tracked"))
    val train8 = getTransformedDataFrame(train7, transform, "signup_app", "signup_app-features", 4).drop(train7.col("signup_app"))
    val train9 = getTransformedDataFrame(train8, transform, "first_device_type", "first_device_type-features", 9).drop(train8.col("first_device_type"))
    val train10 = getTransformedDataFrame(train9, transform, "first_browser", "first_browser-features", 55).drop(train9.col("first_browser"))

    // Join Train and Session
    val sqlContext = new SQLContext(sc)
    val sessions = sqlContext.read.load(sessionPath)
    val train_session = train10.join(sessions, train10("id") === sessions("user_id"), joinType = "left_outer")

    val removeNullVector = functions.udf((v: Vector) => {
      if(v == null){
        Vectors.zeros(Helper.sessionAction+Helper.sessiondeviceType)
      }else {
        v
      }
    })

    val train_session_zeros = train_session.withColumn("session-features-sanitized", removeNullVector(train_session.col("session-features"))).drop(train_session.col("session-features"))

    val assembler = new VectorAssembler()
      .setInputCols(Array("signup_method-features", "language_sanitized-features",
                          "affiliate_channel-features", "affiliate_provider-features", "first_affiliate_tracked-features",
                          "signup_app-features", "first_device_type-features", "first_browser-features", "session-features-sanitized"))
      .setOutputCol("features")

    val output = assembler.transform(train_session_zeros)

    Write.csv("/Users/ankushkhanna/Documents/kaggle/airbnb/train_session_csv", output.select("id", "features", "country_destination"))
    output.select("id", "features", "country_destination").coalesce(1).write.mode(SaveMode.Overwrite).save("/Users/ankushkhanna/Documents/kaggle/airbnb/train_session")
  }


  private def getTransformedDataFrame(data: DataFrame, transform: Transform, inputCol: String, outputCol: String, noOfFeatures: Int): DataFrame = {
    val pipelineAction = new Pipeline().setStages(transform.apply(Array(), inputCol, outputCol, noOfFeatures)._1)
    val modelAction = pipelineAction.fit(data)
    modelAction.transform(data)
  }
}
