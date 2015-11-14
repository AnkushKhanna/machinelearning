package common.evaluator

import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.DoubleType

import scala.math.log

class MultiClassLogLoss(override val uid: String) extends Evaluator {

  val labelCol = new Param[String](this, "inputCol", "input column")
  val predictionCol = new Param[String](this, "outputCol", "output column")
  val probabilityCol = new Param[String](this, "probabilityCol", "probability column")

  type VectorType = org.apache.spark.mllib.linalg.VectorUDT
  val VT = new VectorType

  def this() = this(Identifiable.randomUID("mcLogLoss"))

  def setLabelCol(value: String): this.type = set(labelCol, value)

  def setPredicationCol(value: String): this.type = set(predictionCol, value)

  def setProbabilityCol(value: String): this.type = set(probabilityCol, value)

  override def evaluate(dataset: DataFrame): Double = {
    val n = dataset.count()

    val schema = dataset.schema
    require(schema($(predictionCol)).dataType.equals(DoubleType))
    require(schema($(labelCol)).dataType.equals(DoubleType))
    require(schema($(probabilityCol)).dataType.equals(VT))

    val result = dataset.select($(predictionCol), $(labelCol), $(probabilityCol))
      .map { row =>
        val prob = row.getAs[DenseVector](2)
        1 * log(prob(row.getDouble(1).toInt))
    }

    -result.sum()/n
  }

  override def copy(extra: ParamMap): Evaluator = this

}
