package common.operations

import org.apache.spark.sql.functions

object Clean {
  def remove(value: String) = functions.udf((dd: String) => {
    if (dd == value) "" else dd
  })
}
