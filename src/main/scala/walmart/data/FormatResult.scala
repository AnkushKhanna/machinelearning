package walmart.data

import org.apache.spark.SparkContext
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.{SaveMode, DataFrame, Row, SQLContext}

class FormatResult(sc: SparkContext) extends Serializable {


  def format(predictions: DataFrame) = {



    val brc = sc.broadcast(map)

    val rdds = predictions.select("VisitNumber", "probabilityWithLabel").map(r => {
      val prob = r.getAs[String](1)
      val v = prob.split(", ")
      val a = new Array[String](brc.value.size)
      a.update(0, r.getAs[Integer](0).toString)

      v.foreach(x => {
        val arr = x.split("::")
        a.update(brc.value.get("TripType_" + arr(0)).getOrElse(38), arr(1))
      })
      a.mkString(", ")
    })

    rdds.coalesce(1).saveAsTextFile("src/main/resources/submissionResult")
  }

  def map = Map[String, Int](
    "VisitNumber" -> 0,
    "TripType_3" -> 1,
    "TripType_4" -> 2,
    "TripType_5" -> 3,
    "TripType_6" -> 4,
    "TripType_7" -> 5,
    "TripType_8" -> 6,
    "TripType_9" -> 7,
    "TripType_12" -> 8,
    "TripType_14" -> 9,
    "TripType_15" -> 10,
    "TripType_18" -> 11,
    "TripType_19" -> 12,
    "TripType_20" -> 13,
    "TripType_21" -> 14,
    "TripType_22" -> 15,
    "TripType_23" -> 16,
    "TripType_24" -> 17,
    "TripType_25" -> 18,
    "TripType_26" -> 19,
    "TripType_27" -> 20,
    "TripType_28" -> 21,
    "TripType_29" -> 22,
    "TripType_30" -> 23,
    "TripType_31" -> 24,
    "TripType_32" -> 25,
    "TripType_33" -> 26,
    "TripType_34" -> 27,
    "TripType_35" -> 28,
    "TripType_36" -> 29,
    "TripType_37" -> 30,
    "TripType_38" -> 31,
    "TripType_39" -> 32,
    "TripType_40" -> 33,
    "TripType_41" -> 34,
    "TripType_42" -> 35,
    "TripType_43" -> 36,
    "TripType_44" -> 37,
    "TripType_999" -> 38)
}
