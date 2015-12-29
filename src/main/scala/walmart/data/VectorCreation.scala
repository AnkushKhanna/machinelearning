package walmart.data

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable.Map

class VectorCreation(sc: SparkContext) {
  def createVector: RDD[LabeledPoint] = {
    val department = new Department(sc)
    val map: Map[String, Int] = department.createDepartment

    val train = sc.textFile("src/main/resources/train.csv")
    val trainData = train.map(_.split(",")).filter(_.length == 7).filter(x => x(0) != """"TripType"""").map(x => new TripData(x))

    val visitToTrainData = trainData.groupBy(td => td.visitNumber)

    val broadcast = sc.broadcast(map)

    // println(map.size)

    val labelPoints = visitToTrainData.map(value => {
      val tds = value._2

      val vectorForNegative = 2

      val vectorArray = new Array[Double](broadcast.value.size * vectorForNegative)
      var tripType = 0.0;
      tds.foreach(td => {
        tripType = td.tripType.toDouble
        val option = broadcast.value.get(td.departmentDescription)
        if (option.isDefined) {
          val index =
            if (td.scanCount.toInt < 0) {
              (option.get * vectorForNegative) - 1
            } else {
              option.get - 1
            }
          vectorArray(index) += td.scanCount.toInt
        }
      })
      LabeledPoint(tripType, Vectors.dense(vectorArray))
    })

    labelPoints
  }
}
