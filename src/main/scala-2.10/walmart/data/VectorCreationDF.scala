package walmart.data

import common.UserDefinedAggregator.{AlwaysFirst, ConcatenateString}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, SQLContext}

object start {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "test")
    val v = new VectorCreationDF(sc)
    v.createTrainVector("src/main/resources/train.csv")
  }
}

class VectorCreationDF(sc: SparkContext) {

  def createTrainVector(path: String): DataFrame = {

    val dataFrame = load(path)

    val concatenate = new ConcatenateString("DepartmentDescription")
    val tripType = new AlwaysFirst("TripType")

    val groupedDataFrame = dataFrame.groupBy("VisitNumber")
      .agg(
        concatenate(dataFrame.col("DepartmentDescription")).as("Agg-DepartmentDescription"),
        tripType(dataFrame.col("TripType")).as("label")
      )

    val features = transform(groupedDataFrame)
    features.select("VisitNumber","label", "features").cache()
  }

  def createTestVector(path: String): DataFrame = {
    val dataFrame = load(path)

    val concatenate = new ConcatenateString("DepartmentDescription")

    val groupedDataFrame = dataFrame.groupBy("VisitNumber")
      .agg(
        concatenate(dataFrame.col("DepartmentDescription")).as("Agg-DepartmentDescription")
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

    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered")

    val ngram = new NGram()
      .setInputCol("words")
      .setOutputCol("ngrams")
      .setN(2)

    val htf = new HashingTF()
      .setInputCol("ngrams")
      .setOutputCol("hash")
      .setNumFeatures(1400)

    val idf = new IDF()
      .setInputCol("hash")
      .setOutputCol("idf-features")

    val normalizer = new Normalizer()
      .setInputCol("idf-features")
      .setOutputCol("features")

    val pca = new PCA()
      .setInputCol("n-features")
      .setOutputCol("features")
      .setK(500)

    val pipeline = new Pipeline().setStages(Array(tokenizer, ngram,  htf, idf, normalizer))
    val model = pipeline.fit(dataFrame)

    model.transform(dataFrame)
  }
}
