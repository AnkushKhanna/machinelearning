package LearningSpark

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.SQLContext

object Learning {
  def main(args: Array[String]) {
    //vectorAssembler
    OneHotEncoder
  }

  val sqlContext = new SQLContext(new SparkContext("local[1]", "test"))

  def polyExp = {
    val data = Array(
      Vectors.dense(-2.0, 2.3),
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.6, -1.1)
    )
    val df = sqlContext.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    val polynomialExpansion = new PolynomialExpansion()
      .setInputCol("features")
      .setOutputCol("polyFeatures")
      .setDegree(2)
    val polyDF = polynomialExpansion.transform(df)
    polyDF.select("polyFeatures").take(3).foreach(println)
  }

  def DCT = {
    val data = Seq(
      Vectors.dense(0.0, 1.0, -2.0, 3.0),
      Vectors.dense(-1.0, 2.0, 4.0, -7.0),
      Vectors.dense(14.0, -2.0, -5.0, 1.0))
    val df = sqlContext.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    val dct = new DCT()
      .setInputCol("features")
      .setOutputCol("featuresDCT")
      .setInverse(false)
    val dctDf = dct.transform(df)
    dctDf.select("featuresDCT").take(3).foreach(println)
  }

  def OneHotEncoder = {
    val df = sqlContext.createDataFrame(Seq(
      (0, "a dsd sdsd"),
      (1, "a dsd sdsd"),
      (2, "c fff"),
      (3, "a dfdd"),
      (4, "a fff"),
      (5, "c jjjj")
    )).toDF("id", "category")

    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)
    val indexed = indexer.transform(df)

    val encoder = new OneHotEncoder().setInputCol("categoryIndex").
      setOutputCol("categoryVec")
    val encoded = encoder.transform(indexed)
    println (encoded.head().getAs[Vector]("categoryVec").size)
    encoded.select("id", "categoryVec").foreach(println)
  }

  def bucketizer() = {
    val splits = Array(Double.NegativeInfinity, -0.5, -0.35, 0.0, 0.5, Double.PositiveInfinity)

    val data = Array(-0.5, -0.3, 0.0, 0.2)
    val dataFrame = sqlContext.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    dataFrame.groupBy()

    val bucketizer = new Bucketizer()
      .setInputCol("features")
      .setOutputCol("bucketedFeatures")
      .setSplits(splits)

    // Transform original data into its bucket index.
    val bucketedData = bucketizer.transform(dataFrame)
    bucketedData.select("bucketedFeatures").take(4).foreach(println)
  }

  def vectorAssembler() = {
    val dataset = sqlContext.createDataFrame(
      Seq((0, 18, 1.0, Vectors.dense(0.0, 10.0, 0.5),  Vectors.dense(20.0, 20.0, 20.5), 1.0))
    ).toDF("id", "hour", "mobile", "userFeatures", "userFeatures2", "clicked")
    val assembler = new VectorAssembler()
      .setInputCols(Array("hour", "mobile", "userFeatures", "userFeatures2"))
      .setOutputCol("features")
    val output = assembler.transform(dataset)
    println(output.select("features", "clicked").first())
  }

  def ployExpan = {
    val data = Array(
      Vectors.dense(-2.0, 2.3),
      Vectors.dense(0.0, 0.0),
      Vectors.dense(0.6, -1.1)
    )
    val df = sqlContext.createDataFrame(data.map(Tuple1.apply)).toDF("features")
    val polynomialExpansion = new PolynomialExpansion()
      .setInputCol("features")
      .setOutputCol("polyFeatures")
      .setDegree(2)
    val polyDF = polynomialExpansion.transform(df)
    polyDF.select("polyFeatures").take(3).foreach(println)
  }
}
