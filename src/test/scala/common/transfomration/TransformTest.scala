package common.transfomration

import junit.framework.Assert
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class TransformTest extends FunSuite with BeforeAndAfterAll {

  private val master = "local[1]"
  private val appName = "test-transform"

  private var sqlContext: SQLContext = _
  private var sc: SparkContext = _
  private var data: DataFrame = _

  override protected def beforeAll {
    val conf = new SparkConf()
      .setMaster(master)
      .setAppName(appName)

    sc = new SparkContext(conf)
    sqlContext = new SQLContext(sc)

    data = sqlContext.createDataFrame(Seq(
      ("0", "lookup Windows Desktop"),
      ("1", "lookup Windows Desktop"),
      ("2", "search_results click view_search_results Windows Desktop"),
      ("3", "lookup Windows Desktops")
    )).toDF("id", "action")
  }

  override protected def afterAll {
    if (sc != null) {
      sc.stop()
    }
  }

  test("testing tokenizer and hashing transformation") {

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

  test("testing tokenizer transformation") {

    val transform = new Transform with TTokenize
    val pipelineAction = new Pipeline().setStages(transform.apply(Array(), "action", "action_features", 50)._1)
    val modelAction = pipelineAction.fit(data)
    val features = modelAction.transform(data)

    Assert.assertEquals(4, features.count())
    Assert.assertEquals(features.filter(features("id") === "0").head().getAs[Vector]("action_features"),
      features.filter(features("id") === "1").head().getAs[Vector]("action_features"))
    Assert.assertNotSame(features.filter(features("id") === "0").head().getAs[Vector]("action_features"),
      features.filter(features("id") === "3").head().getAs[Vector]("action_features"))
  }

  test("testing tokenizer, hashing and idf transformation") {

    val transform = new Transform with TTokenize with THashing with TIDF
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
