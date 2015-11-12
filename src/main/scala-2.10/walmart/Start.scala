package walmart

import common.ml.RandomForest
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import walmart.data.{VectorConversionToRdd, FormatResult, VectorCreationDF}
import walmart.evaluation.WalmartEvaluator


object Start {
  def main(args: Array[String]) {

    val conf = new SparkConf() //
      .set("spark.driver.maxResultSize", "2g")
      //.set("spark.executor.memory", "6g")
      .setAppName("walmart")
      .setMaster("local[5]")
      .set("spark.driver.cores", "4")

    val runningFinal = true

    val sc = new SparkContext(conf)

    val Array(trainingData, testData) =
      if (runningFinal) {
        Array(new VectorCreationDF(sc).createTrainVector("src/main/resources/train.csv"),
          new VectorCreationDF(sc).createTestVector("src/main/resources/test.csv"))
      } else {
        val train = new VectorCreationDF(sc).createTrainVector("src/main/resources/train.csv")
        train.randomSplit(Array(0.7, 0.3))
      }

    val conversion = new VectorConversionToRdd(sc)
    val trainRDD = conversion.convert(trainingData)
    val testRDD = conversion.convert(testData)

    println(trainRDD.count, testRDD.count)

//    val sqlContext = new SQLContext(sc)
//
//
//    val decisionTree = new RandomForest(sc)
//    val model = decisionTree.fit(sqlContext.createDataFrame(trainRDD).toDF("label", "features"))
//
//    val predictions = model.transform(sqlContext.createDataFrame(testRDD).toDF("label", "features"))
//
//    if (!runningFinal) {
//      val evaluator = new WalmartEvaluator
//      evaluator.evaluate(predictions)
//    } else {
//      new FormatResult(sc).format(predictions)
//    }
  }


}
