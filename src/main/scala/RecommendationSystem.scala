import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import akka.stream.ActorMaterializer
import akka.http.scaladsl.server.Directives._
import akka.http.scaladsl.server.Route
import org.apache.spark.sql.functions.explode

import scala.concurrent.ExecutionContextExecutor
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.types._

object RecommendationSystem {

  case class BookData(ISBN: String, title: String, author: String, year: Int, publisher: String, urlS: String, urlM: String, urlL: String)

  case class RatingData(userId: Int, ISBN: Int, rating: Int)

  case class Recommendation(userId: Int, recommendedBook: Int)

  implicit val system: ActorSystem = ActorSystem("RecommendationSystem")
  implicit val materializer: ActorMaterializer = ActorMaterializer()
  implicit val executionContext: ExecutionContextExecutor = system.dispatcher
  def main(args: Array[String]): Unit = {


    val spark = SparkSession.builder()
      .appName("RecommendationSystem")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val (bookDf, ratingDf, jdf) = readAndJoinData(spark)
    val Array(trainingData, testData) = splitData(jdf)
    val alsModel = trainALSModel(trainingData)
    evaluateALSModel(alsModel, testData)

    val route = createAkkaHttpRoute(alsModel, bookDf, spark)
    startAkkaHttpServer(route, system)

    spark.stop()
  }

  def readAndJoinData(spark: SparkSession): (DataFrame, DataFrame, DataFrame) = {
    val bookSchema = StructType(
      StructField("ISBN", StringType, nullable = true) ::
        StructField("Book-Title", StringType, nullable = true) ::
        StructField("Book-Author", StringType, nullable = true) ::
        StructField("Year-Of-Publication", IntegerType, nullable = true) ::
        StructField("Publisher", StringType, nullable = true) ::
        StructField("Image-URL-S", StringType, nullable = true) ::
        StructField("Image-URL-M", StringType, nullable = true) ::
        StructField("Image-URL-L", StringType, nullable = true) ::
        Nil
    )

    val ratingSchema = StructType(
      StructField("USER-ID", IntegerType, nullable = true) ::
        StructField("ISBN", IntegerType, nullable = true) ::
        StructField("Book-Rating", IntegerType, nullable = true) ::
        Nil
    )

    val options = Map("header" -> "true", "delimiter" -> ";", "mode" -> "DROPMALFORMED")

    val bookDf = getDataFrame(spark, "/books_data/Books.csv", options, bookSchema)
    val ratingDf = getDataFrame(spark, "/books_data/Ratings.csv", options, ratingSchema)
    import spark.implicits._

    val jdf = ratingDf.alias("ratings").join(bookDf.alias("books"), $"ratings.ISBN" === $"books.ISBN")
      .select(
        $"ratings.USER-ID".as("userId"),
        $"ratings.Book-Rating".as("rating"),
        $"ratings.ISBN".as("isbn"),
        $"books.Book-Title".as("title"),
        $"books.Book-Author".as("author"),
        $"books.Year-Of-Publication".as("year"),
        $"books.Publisher".as("publisher"),
        $"books.Image-URL-S".as("Image-URL-S"),
        $"books.Image-URL-M".as("Image-URL-M"),
        $"books.Image-URL-L".as("Image-URL-L")
      )

    (bookDf, ratingDf, jdf)
  }


  def splitData(jdf: DataFrame): Array[DataFrame] = {
    jdf.randomSplit(Array(0.8, 0.2))
  }

  def trainALSModel(trainingData: DataFrame): ALSModel = {
    val als = new ALS()
      .setMaxIter(15)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("isbn")
      .setRatingCol("rating")

    als.fit(trainingData)
  }

  def evaluateALSModel(alsModel: ALSModel, testData: DataFrame): Unit = {
    alsModel.setColdStartStrategy("drop")
    val predictions = alsModel.transform(testData)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE): $rmse")
  }

  def getDataFrame(spark: SparkSession, fileSrc: String, conf: Map[String, String], schema: StructType): DataFrame = {
    spark.read.options(conf).schema(schema).csv(RecommendationSystem.getClass.getResource(fileSrc).getPath).toDF()
  }

  def createAkkaHttpRoute(alsModel: ALSModel, bookDf: DataFrame, spark: SparkSession): Route = {
    pathPrefix("recommendations") {
      concat(
        path("user" / IntNumber) { userId =>
          completeRecommendations(userId, alsModel, bookDf, spark);
        },
        path("userchoice" / IntNumber) { userId =>
          completeUserChoice(userId, bookDf, spark)
        }
      )
    }
  }

  def completeRecommendations(userId: Int, alsModel: ALSModel, bookDf: DataFrame, spark: SparkSession): Route = {
    import spark.implicits._
    val userRecommendations = alsModel.recommendForUserSubset(Seq(userId).toDF("userId"), 10)
      .select($"userId", explode($"recommendations").as("rec"))
      .select($"userId", $"rec.ISBN".as("recommendedBook"))

    val recommendedBooks = userRecommendations
      .join(bookDf, userRecommendations("recommendedBook") === bookDf("ISBN"))
      .select($"userId", $"recommendedBook", $"Book-Title", $"Book-Author", $"Year-Of-Publication", $"Publisher", $"Image-URL-S", $"Image-URL-M", $"Image-URL-L")
      .limit(10)
      .collect()

    completeHtmlTable(userId, recommendedBooks)
  }

  def completeUserChoice(userId: Int, bookDf: DataFrame, spark: SparkSession): Route = {
    import spark.implicits._
    val userRatings = bookDf.filter($"userId" === userId)
    val userHighestRatedBooks = userRatings.sort($"rating".desc).limit(10)
    completeHtmlTable(userId, userHighestRatedBooks.collect())
  }

  def completeHtmlTable(userId: Int, data: Array[Row]): Route = {
    val tableHeaders = Seq("User ID", "Recommended Book", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-S", "Image-URL-M", "Image-URL-L")

    val tableRows = data.map { row =>
      val rowData = Seq(
        row.getAs[Int]("userId").toString,
        row.getAs[Int]("recommendedBook").toString,
        row.getAs[String]("Book-Title"),
        row.getAs[String]("Book-Author"),
        row.getAs[Int]("Year-Of-Publication").toString,
        row.getAs[String]("Publisher"),
        row.getAs[String]("Image-URL-S"),
        row.getAs[String]("Image-URL-S"),
        row.getAs[String]("Image-URL-S")
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