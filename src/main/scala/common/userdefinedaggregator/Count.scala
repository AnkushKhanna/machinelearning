package common.userdefinedaggregator

import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.{MutableAggregationBuffer, UserDefinedAggregateFunction}
import org.apache.spark.sql.types._

class Count(column: String) extends UserDefinedAggregateFunction{
  override def inputSchema: StructType = StructType(
    StructField(column, StringType) :: Nil
  )

  override def bufferSchema: StructType = StructType(
    StructField("count", IntegerType) :: Nil
  )

  override def deterministic: Boolean = true

  override def evaluate(buffer: Row): Any = buffer.getAs[Int](0)

  override def dataType: DataType = IntegerType

  override def initialize(buffer: MutableAggregationBuffer): Unit = {
    buffer.update(0, 0)
  }

  override def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
      buffer(0) = buffer.getAs[Int](0) + 1
  }

  override def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
    buffer1(0) = buffer1.getAs[Int](0) + buffer2.getAs[Int](0)
  }
}
