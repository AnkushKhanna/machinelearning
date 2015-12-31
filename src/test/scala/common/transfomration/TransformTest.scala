package common.transfomration

import common.userdefinedaggregator.ConcatenateMultipleColumn
import junit.framework.Assert
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.types.{StringType, StructField}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.scalatest.{BeforeAndAfter, FunSuite}

class TransformTest extends FunSuite with BeforeAndAfter {

  private val master = "local[1]"
  private val appName = "test-transform     "

  private var sqlContext: SQLContext = _
  private var sc: SparkContext = _

  before {
    val conf = new SparkConf()
      .setMaster(master)
      .setAppName(appName)

    sc = new SparkContext(conf)
    sqlContext = new SQLContext(sc)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

  test("testing tokenizer and hashing transformation") {

    val data = sqlContext.createDataFrame(Seq(
      ("0", "lookup Windows Desktop"),
      ("1", "lookup Windows Desktop"),
      ("2", "search_results click view_search_results Windows Desktop"),
      ("3", "lookup Windows Desktops")
    )).toDF("id", "action")

    val transform = new Transform with TTokenize with THashing
    val pipelineAction = new Pipeline().setStages(transform.apply(Array(), "action", "action_features", 50)._1)
    val modelAction = pipelineAction.fit(data)
    val features = modelAction.transform(data)

    Assert.assertEquals(4, features.count())
    Assert.assertEquals(features.filter(features("id") === "0").head().getAs[Vector]("action_features"),
      features.filter(features("id") === "1").head().getAs[Vector]("action_features"))
    Assert.assertNotSame(features.filter(features("id") === "0").head().getAs[Vector]("action_features"),
      features.filter(features("id") === "3").head().getAs[Vector]("action_features"))
  }
}
