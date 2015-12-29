package common.transfomration

import org.apache.spark.ml.{PipelineStage}
import org.apache.spark.ml.feature.{IDF, HashingTF, Tokenizer}


class Transform {
  def apply(array: Array[PipelineStage],
                inputCol: String,
                outputCol: String,
                numFeature: Int = new HashingTF().getNumFeatures): (Array[PipelineStage], String) = {
    (array, inputCol)
  }
}

trait TTokenize extends Transform {
  override def apply(array: Array[PipelineStage], inputCol: String, outputCol: String, numFeature: Int = new HashingTF().getNumFeatures): (Array[PipelineStage], String) = {
    val tempInpColumn = System.currentTimeMillis().toString
    val pair = super.apply(array, inputCol, tempInpColumn, numFeature)
    val tokenizerFC = new Tokenizer()
      .setInputCol(pair._2)
      .setOutputCol(outputCol)
    (pair._1 :+ tokenizerFC, outputCol)
  }
}

trait THashing extends Transform {

  override def apply(array: Array[PipelineStage], inputCol: String, outputCol: String, numFeature: Int = new HashingTF().getNumFeatures): (Array[PipelineStage], String) = {
    val tempInpColumn = System.currentTimeMillis().toString
    val pair = super.apply(array, inputCol, tempInpColumn, numFeature)
    val tokenizerFC = new HashingTF()
      .setInputCol(pair._2)
      .setOutputCol(outputCol)
      .setNumFeatures(numFeature)
    (pair._1 :+ tokenizerFC, outputCol)
  }
}

trait TIDF extends Transform {

  override def apply(array: Array[PipelineStage], inputCol: String, outputCol: String, numFeature: Int = new HashingTF().getNumFeatures): (Array[PipelineStage], String) = {
    val tempInpColumn = System.currentTimeMillis().toString +1
    val pair = super.apply(array, inputCol, tempInpColumn, numFeature)
    val tokenizerFC = new IDF()
      .setInputCol(pair._2)
      .setOutputCol(outputCol)
    (pair._1 :+ tokenizerFC, outputCol)
  }
}
