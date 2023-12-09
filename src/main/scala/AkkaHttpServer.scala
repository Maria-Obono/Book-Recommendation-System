import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import akka.stream.ActorMaterializer
import akka.http.scaladsl.server.Directives._
import akka.http.scaladsl.server.Route
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.Dataset
import scala.concurrent.ExecutionContextExecutor
import RecommendationSystem.{BookData, Recommendation}

trait AkkaHttpServer {
  implicit val system: ActorSystem
  implicit val materializer: ActorMaterializer
  implicit val executionContext: ExecutionContextExecutor

  def createAkkaHttpRoute(alsModel: ALSModel, data: (DataFrame, DataFrame), spark: SparkSession): Route = {
    pathPrefix("recommendations") {
      concat(
        path("user" / IntNumber) { userId =>
          completeRecommendations(userId, alsModel, data._1, spark);
        },
        path("userchoice" / IntNumber) { userId =>
          completeUserChoice(userId, data._2, spark)
        }
      )
    }
  }

  def completeRecommendations(userId: Int, alsModel: ALSModel, bookDf: DataFrame, spark: SparkSession): Route = {
    import spark.implicits._

    val userRecommendations: DataFrame = alsModel.recommendForUserSubset(Seq(userId).toDF("userId"), 10)
      .select($"userId", explode($"recommendations").as("rec"))
      .select($"userId", $"rec.ISBN".as("recommendedBook")).as[Recommendation].toDF()
    val recommendedBooks = userRecommendations
      .join(bookDf, userRecommendations("recommendedBook") === bookDf("ISBN"))
      .select($"ISBN", $"Book-Title".as("title"), $"Book-Author".as("author"), $"Year-Of-Publication".as("year"), $"Publisher".as("publisher"), $"Image-URL-S".as("urlS"), $"Image-URL-M".as("urlM"), $"Image-URL-L".as("urlL"))
      .as[BookData]
      .collect()
    completeHtmlTable(userId, recommendedBooks)
  }

  def completeUserChoice(userId: Int, jdf: DataFrame, spark: SparkSession): Route = {
    import spark.implicits._
    val userRatings = jdf.filter($"userId" === userId)
    val userHighestRatedBooks = userRatings.sort($"rating".desc).limit(10).as[BookData]
    completeHtmlTable(userId, userHighestRatedBooks.collect())
  }


  def completeHtmlTable(userId: Int, data: Array[BookData]): Route = {
    val tableHeaders = Seq("User ID", "Recommended Book", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-S", "Image-URL-M", "Image-URL-L")

    val tableRows = data.map { row =>
      val rowData = Seq(
        userId.toString,
        row.ISBN,
        row.title,
        row.author,
        row.year.toString,
        row.publisher,
        row.urlS,
        row.urlM,
        row.urlL
      )
      s"<tr>${rowData.map(data => s"<td>$data</td>").mkString}</tr>"
    }.mkString

    val htmlTable =
      s"""
         |<html>
         |<head><title>Book Recommendation System</title></head>
         |<body>
         |<h1>Top 10 Book Recommendations for User $userId</h1>
         |<table border="1">
         |  <tr>${tableHeaders.map(header => s"<th>$header</th>").mkString}</tr>
         |  $tableRows
         |</table>
         |</body>
         |</html>
         |""".stripMargin

    complete(HttpEntity(ContentTypes.`text/html(UTF-8)`, htmlTable))
  }

  def startAkkaHttpServer(route: Route, system: ActorSystem): Unit = {
    implicit val system: ActorSystem = ActorSystem("RecommendationSystem")
    implicit val materializer: ActorMaterializer = ActorMaterializer()
    implicit val executionContext: ExecutionContextExecutor = system.dispatcher
    val bindingFuture = Http().bindAndHandle(route, "localhost", 8080)
    println(s"Server online at http://localhost:8080/\nPress RETURN to stop...")
    scala.io.StdIn.readLine()
    bindingFuture
      .flatMap(_.unbind())
      .onComplete(_ => system.terminate())
  }
}
