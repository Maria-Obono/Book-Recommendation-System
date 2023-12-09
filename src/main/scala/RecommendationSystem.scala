import akka.actor.ActorSystem
import akka.stream.ActorMaterializer
import scala.concurrent.ExecutionContextExecutor
import org.apache.spark.sql.{SparkSession}

object RecommendationSystem extends AkkaHttpServer with DataProcessingAndTraining {

  case class BookData(ISBN: String, title: String, author: String, year: Int, publisher: String, urlS: String, urlM: String, urlL: String)

  case class Recommendation(userId: Int, recommendedBook: Int)

  implicit val system: ActorSystem = ActorSystem("RecommendationSystem")
  implicit val materializer: ActorMaterializer = ActorMaterializer()
  implicit val executionContext: ExecutionContextExecutor = system.dispatcher
  def main(args: Array[String]): Unit = {


    val spark = SparkSession.builder()
      .appName("RecommendationSystem")
      .master("local[*]")
      .getOrCreate()


    val (bookDf, jdf) = readAndJoinData(spark)
    val Array(trainingData, testData) = splitData(jdf)
    val alsModel = trainALSModel(trainingData)
    evaluateALSModel(alsModel, testData)

    val route = createAkkaHttpRoute(alsModel, (bookDf, jdf), spark)
    startAkkaHttpServer(route, system)

    spark.stop()
  }
}