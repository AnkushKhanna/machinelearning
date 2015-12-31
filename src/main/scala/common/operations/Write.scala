package common.operations

import org.apache.spark.sql.{DataFrame, SaveMode}

object Write {

  def csv(path: String, df: DataFrame) = {
    df.coalesce(1)
      .write.mode(SaveMode.Overwrite)
      .format("com.databricks.spark.csv")
      .option("header", "true")
      .save(path)
  }
}
