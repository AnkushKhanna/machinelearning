package common.userdefinedaggregator

import junit.framework.Assert
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.types.{StringType, StructField}
import org.apache.spark.{SparkConf, SparkContext}
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{BeforeAndAfter, FunSuite}

@RunWith(classOf[JUnitRunner])
class ConcatenateMultipleColumnTest extends FunSuite with BeforeAndAfter {

  private val master = "local[1]"
  private val appName = "test-multiple-Column"

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
  test("concatenate multiple columns concatenate correctly") {

    val sessions = sqlContext.createDataFrame(Seq(
      ("d1mm9tcy42", "lookup", "", "", "Windows Desktop", 319.0),
      ("d1mm9tcy42", "lookup", "click", "", "Windows Desktop", 683.0),
      ("d1mm9tcy42", "search_results", "click", "view_search_results", "Windows Desktop", 59274.0),
      ("d1mm9tcy42", "lookup", "", "", "Windows Desktop", 95.0)
    )).toDF("user_id", "action", "action_type", "action_detail", "device_type", "secs_elapsed")

    val concAction = new ConcatenateMultipleColumn(
      StructField("action", StringType) ::
        StructField("action_type", StringType) ::
        StructField("action_detail", StringType) :: Nil)

    val groupedSessions = sessions.groupBy("user_id").agg(
      concAction(sessions.col("action"), sessions.col("action_type"), sessions.col("action_detail")).as("agg_action")
    )

    Assert.assertEquals(1, groupedSessions.count())
    Assert.assertEquals("d1mm9tcy42", groupedSessions.head().getAs[String]("user_id"))
    Assert.assertEquals("_lookup__ _lookup__ _lookup_click_ _search_results_click_view_search_results", groupedSessions.head().getAs[String]("agg_action"))
  }
}


