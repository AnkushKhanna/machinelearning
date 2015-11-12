package common.mllib

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.rdd.RDD

class RandomForest {

  def createModel(data: RDD[LabeledPoint], map: Map[String, Int]) = {
    val numClasses = map.size - 1
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 5 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 5
    val maxBins = 32

    RandomForest.trainClassifier(data, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)
  }

}
