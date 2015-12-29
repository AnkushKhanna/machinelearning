package common.transfomration

import org.apache.spark.ml.{PipelineStage}
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}


class Transform {
  def transform(array: Array[PipelineStage],
                inputCol: String,
                outputCol: String,
                numFeature: Int = new HashingTF().getNumFeatures): (Array[PipelineStage], String) = {
    (array, inputCol)
  }
}

trait TTokenize extends Transform {
  override def transform(array: Array[PipelineStage], inputCol: String, outputCol: String, numFeature: Int = new HashingTF().getNumFeatures): (Array[PipelineStage], String) = {
    val tempInpColumn = System.currentTimeMillis().toString
    val pair = super.transform(array, inputCol, tempInpColumn)
    val tokenizerFC = new Tokenizer()
      .setInputCol(pair._2)
      .setOutputCol(outputCol)
    (pair._1 :+ tokenizerFC, outputCol)
  }
}

trait THashing extends Transform {

  override def transform(array: Array[PipelineStage], inputCol: String, outputCol: String, numFeature: Int = new HashingTF().getNumFeatures): (Array[PipelineStage], String) = {
    val tempInpColumn = System.currentTimeMillis().toString
    val pair = super.transform(array, inputCol, tempInpColumn)
    val tokenizerFC = new HashingTF()
      .setInputCol(pair._2)
      .setOutputCol(outputCol)
      .setNumFeatures(numFeature)
    (pair._1 :+ tokenizerFC, outputCol)
  }
}
