package common.transformer

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.{Attribute, NominalAttribute}
import org.apache.spark.ml.param.{Param, ParamMap, Params, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.{Row, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

class IndexToOrderProbability(override val uid: String) extends Transformer{

  final val labels: StringArrayParam = new StringArrayParam(this, "labels",
    "array of labels, if not provided metadata from inputCol is used instead.")
  setDefault(labels, Array.empty[String])

  val inputCol = new Param[String](this, "inputCol", "input column")
  val outputCol = new Param[String](this, "outputCol", "output column")

  type VectorType = org.apache.spark.mllib.linalg.VectorUDT
  val VT = new VectorType

  def this() = this(Identifiable.randomUID("IndexToOrderProbability"))

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)

  def setLabels(value: Array[String]): this.type = set(labels, value)

  def transform(dataset: DataFrame): DataFrame = {

    val inputColSchema = dataset.schema($(inputCol))
    val values = if ($(labels).isEmpty) {
      Attribute.fromStructField(inputColSchema)
        .asInstanceOf[NominalAttribute].values.get
    } else {
      $(labels)
    }

    val orderer = udf { prob: Vector =>
      val builder = new StringBuilder
      for (i <- 0 until values.size) {
        builder.append(values(i) + "::" + prob(i))
        if (values.size - 1 != i) {
          builder.append(", ")
        }
      }
      builder.toString
    }

    dataset.select(col("*"),
      orderer(dataset($(inputCol))).as($(outputCol)))
  }

  override def copy(extra: ParamMap): Transformer = this

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    StructType(schema.fields ++ Seq(StructField($(outputCol), StringType, true)))
  }
}
