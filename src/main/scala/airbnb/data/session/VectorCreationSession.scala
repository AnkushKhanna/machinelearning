package airbnb.data.session

import airbnb.Helper
import common.userdefinedaggregator.{Count, ConcatenateMultipleColumn, ReturnFirst}
import common.operations.{Clean, Read, Write}
import common.transfomration.{THashing, TIDF, TTokenize, Transform}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{Normalizer, MinMaxScaler, StandardScaler, VectorAssembler}
import org.apache.spark.sql.{SaveMode, functions}
import org.apache.spark.sql.types.{StringType, StructField}

class VectorCreationSession(sc: SparkContext) {

  def createSessionVector(path: String, inputPath: String, outputPath: String) = {
    val sessions = Read.csv(path + inputPath, sc)

    val sessions1 = sessions.withColumn("action_sanitized", Clean.remove("-unknown-", "-1")(sessions.col("action"))).drop(sessions.col("action"))
      .withColumn("action_type_sanitized", Clean.remove("-unknown-", "-1")(sessions.col("action_type"))).drop(sessions.col("action_type"))
      .withColumn("action_detail_sanitized", Clean.remove("-unknown-", "-1")(sessions.col("action_detail"))).drop(sessions.col("action_detail"))

    val concAction = new ConcatenateMultipleColumn(
      StructField("action_sanitized", StringType) ::
        StructField("action_type_sanitized", StringType) ::
        StructField("action_detail_sanitized", StringType) :: Nil)

    val deviceType = new ReturnFirst("device_type")

    val count = new Count("device_type")

    val groupedSessions = sessions1.groupBy("user_id").agg(
      concAction(sessions1.col("action_sanitized"), sessions1.col("action_type_sanitized"), sessions1.col("action_detail_sanitized")).as("agg_action"),
      deviceType(sessions1.col("device_type")).as("agg_device_first"),
      count(sessions1.col("device_type")).as("count")
    )

    val transformAction = new Transform with TTokenize with THashing with TIDF

    val pipelineAction = new Pipeline().setStages(transformAction.apply(Array(), "agg_action", "action-features", Helper.sessionAction)._1)
    val modelAction = pipelineAction.fit(groupedSessions)
    val dataAction = modelAction.transform(groupedSessions)

    val transformDevice = new Transform with TTokenize with THashing

    val pipelineDevice = new Pipeline().setStages(transformDevice.apply(Array(), "agg_device_first", "device-features", Helper.sessiondeviceType)._1)
    val modelDevice = pipelineDevice.fit(dataAction)
    val dataDevice = modelDevice.transform(dataAction)

    val assembler = new VectorAssembler()
      .setInputCols(Array("action-features", "device-features", "count"))
      .setOutputCol("session-features")

    val vectorAssembler = assembler.transform(dataDevice)

    val scaler = new Normalizer()
      .setInputCol("session-features")
      .setOutputCol("scaled-session-features")

    val output = scaler.transform(vectorAssembler)

    Write.csv(path + outputPath+".csv", output.select("user_id", "session-features", "scaled-session-features"))
    output.coalesce(1).select("user_id", "session-features").write.mode(SaveMode.Overwrite).save(path + outputPath)
  }
}
