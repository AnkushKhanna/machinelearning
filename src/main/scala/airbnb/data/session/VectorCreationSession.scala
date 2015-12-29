package airbnb.data.session

import common.UserDefinedAggregator.{ConcatenateMultipleColumn, ReturnFirst}
import common.operations.{Read, Write}
import common.transfomration.{THashing, TIDF, TTokenize, Transform}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.types.{StringType, StructField}

class VectorCreationSession(sc: SparkContext) {

  def createSessionVector(path: String, inputPath: String, outputPath: String) = {
    val sessions = Read.csv(path + inputPath, sc)

    val concAction = new ConcatenateMultipleColumn(
      StructField("action", StringType) ::
        StructField("action_type", StringType) ::
        StructField("action_detail", StringType) :: Nil)

    val deviceType = new ReturnFirst("device_type")

    val groupedSessions = sessions.groupBy("user_id").agg(
      concAction(sessions.col("action"), sessions.col("action_type"), sessions.col("action_detail")).as("agg_action"),
      deviceType(sessions.col("device_type")).as("agg_device_first")
    )

    val transformAction = new Transform with TTokenize with THashing with TIDF

    val pipelineAction = new Pipeline().setStages(transformAction.apply(Array(), "agg_action", "action-features", 200)._1)
    val modelAction = pipelineAction.fit(groupedSessions)
    val dataAction = modelAction.transform(groupedSessions)

    val transformDevice = new Transform with TTokenize with THashing

    val pipelineDevice = new Pipeline().setStages(transformDevice.apply(Array(), "agg_device_first", "device-features", 50)._1)
    val modelDevice = pipelineDevice.fit(dataAction)
    val dataDevice = modelDevice.transform(dataAction)

    val assembler = new VectorAssembler()
      .setInputCols(Array("action-features", "device-features"))
      .setOutputCol("session-features")

    val output = assembler.transform(dataDevice)

    Write.csv(path + outputPath, output.select("user_id", "session-features"))
  }
}
