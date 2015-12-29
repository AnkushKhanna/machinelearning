package common.UserDefinedAggregator

import org.apache.spark.sql.{Column, Row}
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types.{StringType, StructField, StructType, DataType}

class ConcatenateMultipleColumn(columns: Seq[StructField]) extends UserDefinedAggregateFunction {
  override def inputSchema: StructType = StructType(columns)


  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    if (input.getAs[String](0) != null) {
      for (i <- 0 until columns.size) {
        buffer(0) = buffer.getAs[String](0) + "_" +
          input.getAs[String](i).replaceAll(",", "")
      }
      buffer(0) += " "
    }

  }

  override def bufferSchema: StructType = StructType(
    StructField("concatenate", StringType) :: Nil
  )

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    buffer1(0) = buffer1.getAs[String](0) + " " + buffer2.getAs[String](0)
  }

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = ""
  }

  override def deterministic: Boolean = true

  override def evaluate(buffer: Row): Any = buffer.getAs[String](0).split(" ").sortWith(_ < _).mkString(" ").trim

  override def dataType: DataType = StringType
}
