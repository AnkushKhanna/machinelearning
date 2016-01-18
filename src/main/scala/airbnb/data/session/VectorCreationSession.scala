package airbnb.data.session

import airbnb.Helper
import common.userdefinedaggregator.{Count, ConcatenateMultipleColumn, ReturnFirst}
import common.operations.{Clean, Read, Write}
import common.transfomration.{THashing, TIDF, TTokenize, Transform}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{SaveMode, functions}
import org.apache.spark.sql.types.{StringType, StructField}

class VectorCreationSession(sc: SparkContext) {

  def createSessionVector(path: String, inputPath: String, outputPath: String) = {
    val sessions = Read.csv(path + inputPath, sc)

    val splits = Array(Double.NegativeInfinity, 0.1, 60, 300, 600, 1800, 3600, 18000, 36000, 864000, 6048000, Double.PositiveInfinity)

    val bucketizer = new Bucketizer().
      setInputCol("secs_elapsed").
      setOutputCol("buck_secs_elapsed").
      setSplits(splits)

    val sessionSecBuck = bucketizer.transform(sessions)

    val sessions1 = sessionSecBuck.withColumn("action_sanitized", Clean.remove("-unknown-", "-1")(sessionSecBuck.col("action"))).drop(sessionSecBuck.col("action"))
      .withColumn("action_type_sanitized", Clean.remove("-unknown-", "-1")(sessionSecBuck.col("action_type"))).drop(sessionSecBuck.col("action_type"))
      .withColumn("action_detail_sanitized", Clean.remove("-unknown-", "-1")(sessionSecBuck.col("action_detail"))).drop(sessionSecBuck.col("action_detail"))

    val concActionWithBuck = new ConcatenateMultipleColumn(
      StructField("action_sanitized", StringType) ::
        StructField("action_type_sanitized", StringType) ::
        StructField("action_detail_sanitized", StringType) ::
        StructField("buck_secs_elapsed", StringType) :: Nil)

    val concAction = new ConcatenateMultipleColumn(
      StructField("action_sanitized", StringType) ::
        StructField("action_type_sanitized", StringType) ::
        StructField("action_detail_sanitized", StringType) :: Nil)

    val deviceType = new ReturnFirst("device_type")

    val count = new Count("device_type")

    val groupedSessions = sessions1.groupBy("user_id").agg(
      concActionWithBuck(sessions1.col("action_sanitized"), sessions1.col("action_type_sanitized"), sessions1.col("action_detail_sanitized"), sessions1.col("buck_secs_elapsed")).as("agg_action"),
      concAction(sessions1.col("action_sanitized"), sessions1.col("action_type_sanitized"), sessions1.col("action_detail_sanitized")).as("agg_action_wo_buck"),
      deviceType(sessions1.col("device_type")).as("agg_device_first"),
      count(sessions1.col("device_type")).as("count")
    )

    val transformAction = new Transform with TTokenize with THashing with TIDF

    val pipelineAction = new Pipeline().setStages(transformAction.apply(Array(), "agg_action", "action-features", Helper.sessionAction)._1)
    val modelAction = pipelineAction.fit(groupedSessions)
    val dataAction = modelAction.transform(groupedSessions).drop("agg_action")

    val pipelineActionWoBuck = new Pipeline().setStages(transformAction.apply(Array(), "agg_action_wo_buck", "action-features_wo_buck", Helper.sessionAction)._1)
    val modelActionWoBuck = pipelineActionWoBuck.fit(dataAction)
    val dataActionWoBuck = modelActionWoBuck.transform(dataAction).drop("agg_action_wo_buck")

    val transformDevice = new Transform with TTokenize with THashing

    val pipelineDevice = new Pipeline().setStages(transformDevice.apply(Array(), "agg_device_first", "device-features", Helper.sessionDeviceType)._1)
    val modelDevice = pipelineDevice.fit(dataActionWoBuck)
    val dataDevice = modelDevice.transform(dataActionWoBuck).drop("agg_device_first")


    val assembler = new VectorAssembler()
      .setInputCols(Array("action-features", "action-features_wo_buck", "device-features", "count"))
      .setOutputCol("session-features")

    val vectorAssembler = assembler.transform(dataDevice)

    val scaler = new Normalizer()
      .setInputCol("session-features")
      .setOutputCol("scaled-session-features")

    val output = scaler.transform(vectorAssembler)

    //Write.csv(path + outputPath+".csv", output.select("user_id", "session-features", "scaled-session-features"))
    output.coalesce(1).select("user_id", "session-features", "scaled-session-features").write.mode(SaveMode.Overwrite).save(path + outputPath)
  }
}
