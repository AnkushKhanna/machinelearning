package common.ml

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.DataFrame

class OneVsRestC {
  def fit(train: DataFrame, labelCol: String, featureCol: String, indexedLabelCol: String, predictionLabel: String) = {

    val labelIndexer = new StringIndexer()
      .setInputCol(labelCol)
      .setOutputCol(indexedLabelCol)
      .fit(train)

    val ovr = new OneVsRest()
      .setLabelCol(indexedLabelCol)
      .setFeaturesCol(featureCol)
      .setClassifier(new LogisticRegression())

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol(predictionLabel)
      .setLabels(labelIndexer.labels)

    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, ovr, labelConverter))

    // Train model.  This also runs the indexers.
    pipeline.fit(train)
  }
}
