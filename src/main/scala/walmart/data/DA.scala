package walmart.data

import java.io._

import org.apache.spark.SparkContext

import scala.collection.mutable.Map

class Department(sc: SparkContext) {

  def createDepartment : Map[String, Int] = {

    try {
      val is = new ObjectInputStream(new FileInputStream("src/main/resources/Department"))
      val departmentMap: Map[String, Int] = is.readObject().asInstanceOf[Map[String, Int]]
      departmentMap
    } catch  {
      case fnf: FileNotFoundException => createDepartmentList
    }
  }

  private def createDepartmentList: Map[String, Int] = {
    val v = sc.textFile("src/main/resources/train.csv")
    val v1 = v.map(_.split(",")).filter(_.length == 7)
    println(v1.count() + "   " + v.count())
    val v2 = v1.map(_(5)).filter(x => x != """"DepartmentDescription"""").distinct()

    // SMALL LIST LETS DO IT IN NON SPARK WORLD
    val orderedDepartmentList = v2.takeOrdered(v2.count().toInt).toList

    var acc: Int = 0
    val departmentMap = Map[String, Int]()
    for (d <- orderedDepartmentList) {
      acc += 1
      departmentMap += (d -> acc)
    }


    val oos = new ObjectOutputStream(new FileOutputStream("src/main/resources/Department"))
    oos.writeObject(departmentMap)
    oos.flush
    oos.close
    departmentMap
  }
}
