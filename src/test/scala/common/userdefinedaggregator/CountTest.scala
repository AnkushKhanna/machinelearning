package common.userdefinedaggregator

import junit.framework.Assert
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, FunSuite}

class CountTest extends FunSuite with BeforeAndAfterAll {

  private val master = "local[1]"
  private val appName = "test-multiple-Column"

  private var sqlContext: SQLContext = _
  private var sc: SparkContext = _

  override protected def beforeAll {
    val conf = new SparkConf()
      .setMaster(master)
      .setAppName(appName)

    sc = new SparkContext(conf)
    sqlContext = new SQLContext(sc)
  }

  override protected def afterAll {
    if (sc != null) {
      sc.stop()
    }
  }

  test("count while aggregating should do correct count") {

    val sessions = sqlContext.createDataFrame(Seq(
      ("d1mm9tcy42", "lookup", "", "", "Windows Desktop", 319.0),
      ("d1mm9tcy42", "lookup", "click", "", "Windows Desktop", 683.0),
      ("d1mm9tcy42", "search_results", "click", "view_search_results", "Windows Desktop", 59274.0),
      ("d1mm9tcy42", "lookup", "", "", "Windows Desktop", 95.0),
      ("ryxkloom00", "lookup", "", "", "Windows Desktop", 319.0),
      ("ryxkloom00", "lookup", "click", "", "Windows Desktop", 683.0),
      ("ryxkloom00", "search_results", "click", "view_search_results", "Windows Desktop", 59274.0),
      ("ryxkloom00", "lookup", "", "", "Windows Desktop", 95.0),
      ("abbccddee1", "lookup", "", "", "Windows Desktop", 95.0)
    )).toDF("user_id", "action", "action_type", "action_detail", "device_type", "secs_elapsed")

    val countAction = new Count("device_type")
    val groupedSessions = sessions.groupBy("user_id").agg(
      countAction(sessions.col("device_type")).as("count")
    )

    Assert.assertEquals(3, groupedSessions.count())
    val user1 = groupedSessions.filter(groupedSessions("user_id") === "d1mm9tcy42")
    val user2 = groupedSessions.filter(groupedSessions("user_id") === "ryxkloom00")
    val user3 = groupedSessions.filter(groupedSessions("user_id") === "abbccddee1")
    Assert.assertEquals(4, user1.head().getAs[Int]("count"))
    Assert.assertEquals(4, user2.head().getAs[Int]("count"))
    Assert.assertEquals(1, user3.head().getAs[Int]("count"))
  }
}


