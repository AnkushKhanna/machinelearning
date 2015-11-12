package common.ml

import common.transformer.IndexToOrderProbability
import org.apache.spark.SparkContext
import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.DataFrame

class RandomForest(sc: SparkContext) {
   def fit(data: DataFrame): PipelineModel = {
     println("**************************************************************" +
       "DECISION TREE" +
       "********************************************************************")

     val maxBin = 100
     // Index labels, adding metadata to the label column.
     // Fit on whole dataset to include all labels in index.
     val labelIndexer = new StringIndexer()
       .setInputCol("label")
       .setOutputCol("indexedLabel")
       .fit(data)

     // Automatically identify categorical features, and index them.
     val featureIndexer = new VectorIndexer()
       .setInputCol("features")
       .setOutputCol("indexedFeatures")
       .setMaxCategories(maxBin)
       .fit(data)

     // Train a DecisionTree model.
     val dt = new RandomForestClassifier()
       .setLabelCol("indexedLabel")
       .setFeaturesCol("indexedFeatures")
       .setMaxBins(maxBin)
       .setNumTrees(3)
       .setMaxDepth(3)

     // Convert indexed labels back to original labels.
     val labelConverter = new IndexToString()
       .setInputCol("prediction")
       .setOutputCol("predictedLabel")
       .setLabels(labelIndexer.labels)

     // Convert indexed probability back to original probability place.
     val probabilityConverter = new IndexToOrderProbability()
       .setInputCol("probability")
       .setOutputCol("probabilityWithLabel")
       .setLabels(labelIndexer.labels)

     // Chain indexers and tree in a Pipeline
     val pipeline = new Pipeline()
       .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter, probabilityConverter))

     // Train model.  This also runs the indexers.
     pipeline.fit(data)
   }
 }
