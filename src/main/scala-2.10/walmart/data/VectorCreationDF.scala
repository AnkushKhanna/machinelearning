package walmart.data

import java.io.{FileInputStream, ObjectInputStream}

import common.UserDefinedAggregator.{AlwaysFirst, ConcatenateString}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SQLContext}

import scala.collection.mutable.Map

object start {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "test")
    val v = new VectorCreationDF(sc)
    v.createTrainVector("src/main/resources/train.csv")
  }
}

class VectorCreationDF(sc: SparkContext) {

  def addScanCountToDepartmentDesciption(dataFrame: DataFrame) = {

    val add = (dd: String, sc: String) => {
      val ddM =
        if (sc.toInt < 0) "-" + dd
        else dd
      val builder = new StringBuilder(ddM)
      val count = sc.toInt - 1
      for (i <- 0 until count) builder.append(" " + ddM)
      builder.toString
    }

    val addScanCount = udf(add)

    dataFrame.withColumn(
      "DepartmentDescriptionWithCount",
      addScanCount(
        dataFrame.col("DepartmentDescription"),
        dataFrame.col("ScanCount")
      )
    )
  }



  def createTrainVector(path: String): DataFrame = {

    val is = new ObjectInputStream(new FileInputStream("src/main/resources/FineLineBuck"))
    val fineLineMap: Map[Int, Double] = is.readObject().asInstanceOf[Map[Int, Double]]

    val brc = sc.broadcast(fineLineMap);

    val toBucket = udf ((fineline: Int) => {
      brc.value.get(fineline).getOrElse(-1.0)
    })

    val dataFrame = load(path)
    val dataFrameDD = dataFrame.filter(dataFrame("DepartmentDescription") !== "NULL")

    val dataFrameWNull = dataFrame.filter(dataFrame("FinelineNumber").isNotNull)

    val dataFrameDDBucket = dataFrameWNull.withColumn("fineline_bucket", toBucket(dataFrameWNull("FinelineNumber")))

    val concatenate = new ConcatenateString("DepartmentDescription")
    val concatenateFLN = new ConcatenateString("fineline_bucket")
    val tripType = new AlwaysFirst("TripType")
    val dayType = new AlwaysFirst("Weekday")

    val dataFrameCount = addScanCountToDepartmentDesciption(dataFrameDDBucket)

    val groupedDataFrame = dataFrameCount.groupBy("VisitNumber")
      .agg(
        concatenate(dataFrameCount.col("DepartmentDescriptionWithCount")).as("Agg-DepartmentDescription"),
        tripType(dataFrameCount.col("TripType")).as("label"),
        dayType(dataFrameCount.col("Weekday")).as("Day"),
        concatenateFLN(dataFrameCount.col("fineline_bucket")).as("FLN-C")
      )

    val features = transform(groupedDataFrame)
    features.select("VisitNumber", "label", "features").cache()
  }


  def createTestVector(path: String): DataFrame = {
    val dataFrame = load(path)

    val dataFrameDD = dataFrame.filter(dataFrame("DepartmentDescription") !== "NULL")

    val concatenate = new ConcatenateString("DepartmentDescription")
    val dayType = new AlwaysFirst("Weekday")
    val dataFrameCount = addScanCountToDepartmentDesciption(dataFrameDD)

    val groupedDataFrame = dataFrameCount.groupBy("VisitNumber")
      .agg(
        concatenate(dataFrameCount.col("DepartmentDescription")).as("Agg-DepartmentDescription"),
        dayType(dataFrameCount.col("Weekday")).as("Day")
      )

    val features = transform(groupedDataFrame)
    features.select("VisitNumber", "features")
  }

  private def load(path: String): DataFrame = {
    val sqlContext = new SQLContext(sc)

    sqlContext.read
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(path).toDF()
  }

  private def transform(dataFrame: DataFrame): DataFrame = {
    val tokenizer = new Tokenizer()
      .setInputCol("Agg-DepartmentDescription")
      .setOutputCol("words")

    val ngram = new NGram()
      .setInputCol("words")
      .setOutputCol("ngrams")
      .setN(1)

    val htf = new HashingTF()
      .setInputCol("ngrams")
      .setOutputCol("hash")
      .setNumFeatures(1000)

    val idf = new IDF()
      .setInputCol("hash")
      .setOutputCol("dd-features")

//    val normalizer = new Normalizer()
//      .setInputCol("idf-features")
//      .setOutputCol("n-features")
//
//    val pca = new PCA()
//      .setInputCol("idf-features")
//      .setOutputCol("features")
//      .setK(1000)

    val pipeline = new Pipeline().setStages(Array(tokenizer, ngram, htf, idf))
    val model = pipeline.fit(dataFrame)

    val dataSetDD = model.transform(dataFrame)


    val tokenizerW = new Tokenizer()
      .setInputCol("Day")
      .setOutputCol("dayT")

    val htfW = new HashingTF()
      .setInputCol("dayT")
      .setOutputCol("dayVector")
      .setNumFeatures(7)

    val pipelineW = new Pipeline().setStages(Array(tokenizerW, htfW))

    val modelW = pipelineW.fit(dataSetDD)

    val dataSetDDW = modelW.transform(dataSetDD)


    val tokenizerF = new Tokenizer()
      .setInputCol("FLN-C")
      .setOutputCol("FLN-CT")

    val htfF = new HashingTF()
      .setInputCol("FLN-CT")
      .setOutputCol("FLN-features")
      .setNumFeatures(12)

    val idfF = new IDF()
      .setInputCol("FLN-Vector")
      .setOutputCol("FLN-features")

    val pipelineF = new Pipeline().setStages(Array(tokenizerF, htfF))

    val modelF = pipelineF.fit(dataSetDD)

    val dataSetDDF = modelF.transform(dataSetDDW)


    val assembler = new VectorAssembler()
      .setInputCols(Array("dd-features", "dayVector", "FLN-features"))
      .setOutputCol("features")

    assembler.transform(dataSetDDF)

  }
}
