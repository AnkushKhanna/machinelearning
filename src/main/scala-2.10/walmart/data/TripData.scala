package walmart.data

class TripData(data: Array[String]) extends Serializable{

  val tripType = data(0)
  val visitNumber = data(1)
  val weekday = data(2)
  val upc = data(3)
  val scanCount = data(4)
  val departmentDescription = data(5)
  val fineLineNumber = data(6)
}
