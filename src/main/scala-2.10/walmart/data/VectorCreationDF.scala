package walmart.data

import common.UserDefinedAggregator.{AlwaysFirst, ConcatenateString}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame, SQLContext}

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

    val dataFrame = load(path)

    val concatenate = new ConcatenateString("DepartmentDescription")
    val tripType = new AlwaysFirst("TripType")

    val dataFrameCount = addScanCountToDepartmentDesciption(dataFrame)

    val groupedDataFrame = dataFrameCount.groupBy("VisitNumber")
      .agg(
        concatenate(dataFrameCount.col("DepartmentDescriptionWithCount")).as("Agg-DepartmentDescription"),
        tripType(dataFrameCount.col("TripType")).as("label")
      )

    val features = transform(groupedDataFrame)
    features.select("VisitNumber", "label", "features").cache()
  }


  def createTestVector(path: String): DataFrame = {
    val dataFrame = load(path)

    val concatenate = new ConcatenateString("DepartmentDescription")

    val dataFrameCount = addScanCountToDepartmentDesciption(dataFrame)

    val groupedDataFrame = dataFrameCount.groupBy("VisitNumber")
      .agg(
        concatenate(dataFrameCount.col("DepartmentDescription")).as("Agg-DepartmentDescription")
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
      .setNumFeatures(2000)

    val idf = new IDF()
      .setInputCol("hash")
      .setOutputCol("idf-features")

    val normalizer = new Normalizer()
      .setInputCol("idf-features")
      .setOutputCol("n-features")

    val pca = new PCA()
      .setInputCol("idf-features")
      .setOutputCol("features")
      .setK(1000)

    val pipeline = new Pipeline().setStages(Array(tokenizer, ngram, htf, idf, pca))
    val model = pipeline.fit(dataFrame)

    model.transform(dataFrame)
  }
}
