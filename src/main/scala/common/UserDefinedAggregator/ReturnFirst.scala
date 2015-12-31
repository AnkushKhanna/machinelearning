package common.userdefinedaggregator

import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types.{DataType, StringType, StructField, StructType}

class ReturnFirst(column: String)  extends UserDefinedAggregateFunction {
  override def inputSchema: StructType = StructType(
    StructField(column, StringType) :: Nil
  )

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
    buffer(0) = input.getAs[String](0)
  }

  override def bufferSchema: StructType = StructType(
    StructField("visit", StringType) :: Nil
  )

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    buffer1(0) = buffer2.getAs[String](0)
  }

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer(0) = "0"
  }

  override def deterministic: Boolean = true

  override def evaluate(buffer: Row): Any = buffer.getAs[String](0)

  override def dataType: DataType = StringType
}
