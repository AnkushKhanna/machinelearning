package walmart

import common.ml.RandomForest
import org.apache.spark.sql.SaveMode
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
      .set("spark.driver.cores", "3")//.set("spark.sql.tungsten.enabled", "false")

    val runningFinal = false

    val sc = new SparkContext(conf)

    val Array(trainingData, testData) =
      if (runningFinal) {
        Array(new VectorCreationDF(sc).createTrainVector("src/main/resources/train.csv"),
          new VectorCreationDF(sc).createTestVector("src/main/resources/test.csv"))
      } else {
        val train = new VectorCreationDF(sc).createTrainVector("src/main/resources/train.csv")
        train.randomSplit(Array(0.7, 0.3))
      }


    val decisionTree = new RandomForest(sc)
    val model = decisionTree.fit(trainingData, testData)

    // Make predictions.
    val predictions = model.transform(testData)

    if (!runningFinal) {
      val evaluator = new WalmartEvaluator
      evaluator.evaluate(predictions)
    } else {
      //new FormatResult(sc).format(predictions)
      predictions.select("VisitNumber", "probability").coalesce(1)
        .write.mode(SaveMode.Overwrite)
            .format("com.databricks.spark.csv")
            .option("header", "true")
            .save("src/main/resources/submissionResult2")
    }
  }
}
