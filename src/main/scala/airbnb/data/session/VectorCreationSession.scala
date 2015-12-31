package airbnb.data.session

import airbnb.Helper
import common.userdefinedaggregator.{ConcatenateMultipleColumn, ReturnFirst}
import common.operations.{Clean, Read, Write}
import common.transfomration.{THashing, TIDF, TTokenize, Transform}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{SaveMode, functions}
import org.apache.spark.sql.types.{StringType, StructField}

class VectorCreationSession(sc: SparkContext) {

  def createSessionVector(path: String, inputPath: String, outputPath: String) = {
    val sessions = Read.csv(path + inputPath, sc)

    val sessions1 = sessions.withColumn("action_sanitized", Clean.remove("-unknown-")(sessions.col("action"))).drop(sessions.col("action"))
      .withColumn("action_type_sanitized", Clean.remove("-unknown-")(sessions.col("action_type"))).drop(sessions.col("action_type"))
      .withColumn("action_detail_sanitized", Clean.remove("-unknown-")(sessions.col("action_detail"))).drop(sessions.col("action_detail"))

    val concAction = new ConcatenateMultipleColumn(
      StructField("action_sanitized", StringType) ::
        StructField("action_type_sanitized", StringType) ::
        StructField("action_detail_sanitized", StringType) :: Nil)

    val deviceType = new ReturnFirst("device_type")

    val groupedSessions = sessions1.groupBy("user_id").agg(
      concAction(sessions1.col("action_sanitized"), sessions1.col("action_type_sanitized"), sessions1.col("action_detail_sanitized")).as("agg_action"),
      deviceType(sessions1.col("device_type")).as("agg_device_first")
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
      .setInputCols(Array("action-features", "device-features"))
      .setOutputCol("session-features")

    val output = assembler.transform(dataDevice)

    //Write.csv(path + outputPath, output.select("user_id", "session-features"))
    output.coalesce(1).select("user_id", "session-features").write.mode(SaveMode.Overwrite).save(path + outputPath)
  }
}
