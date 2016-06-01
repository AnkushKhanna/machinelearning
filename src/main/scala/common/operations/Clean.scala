package common.operations

import org.apache.spark.sql.functions

object Clean {
  def remove(key: String, value: String) = functions.udf((dd: String) => {
    if (dd == key) value else dd
  })
}
