package walmart

import common.ml.RandomForest
import org.apache.spark.sql.{Column, DataFrame, SaveMode}
import org.apache.spark.{SparkConf, SparkContext}
import walmart.data.{FormatResult, VectorCreationDF}
import walmart.evaluation.WalmartEvaluator


object Start {
  def main(args: Array[String]) {

    val conf = new SparkConf() //
      .set("spark.driver.maxResultSize", "3g")
      //.set("spark.executor.memory", "6g")
      .setAppName("walmart")
      .setMaster("local[2]")
      .set("spark.driver.cores", "3")

    val runningFinal = false

    val sc = new SparkContext(conf)

    val Array(trainingData, testData) =
      if (runningFinal) {
        Array(new VectorCreationDF(sc).createTrainVector("src/main/resources/train.csv"),
          new VectorCreationDF(sc).createTestVector("src/main/resources/test.csv"))
      } else {
        val train = new VectorCreationDF(sc).createTrainVector("src/main/resources/train.csv")
        train.randomSplit(Array(0.7, 0.5))
      }

    //    val vectorCreation = new VectorCreation(sc)
    //    val labelPoints = vectorCreation.createVector
    //    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    //    val data = sqlContext.createDataFrame(labelPoints).toDF("label", "features")
    //

   // val test = testData.drop("label")

    val decisionTree = new RandomForest(sc)
    val model = decisionTree.fit(trainingData)

    testData.coalesce(1).write.format("json")
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .save("src/main/resources/testData")

    // Make predictions.
    val predictions = model.transform(testData)

//    predictions.coalesce(1).select("probability","probabilityWithLabel")
//      .write.format("json")
//      .mode(SaveMode.Overwrite)
//      .option("header", "true")
//      .save("src/main/resources/finalResult2")

    if (!runningFinal) {
      val evaluator = new WalmartEvaluator
      evaluator.evaluate(predictions)
    } else {
      new FormatResult(sc).format(predictions)
    }
  }
}
