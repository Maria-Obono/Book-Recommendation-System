import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import akka.stream.ActorMaterializer
import akka.http.scaladsl.server.Directives._
import org.apache.spark.sql.functions.explode
import scala.concurrent.ExecutionContextExecutor
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType}


object RecommendationSystem {
  def main(args: Array[String]): Unit = {
    implicit val system: ActorSystem = ActorSystem("RecommendationSystem")
    implicit val materializer: ActorMaterializer = ActorMaterializer()
    implicit val executionContext: ExecutionContextExecutor = system.dispatcher

    // Create a SparkSession
    val spark = SparkSession.builder()
      .appName("RecommendationSystem")
      .master("local[*]") // Use local mode; adjust according to your cluster setup
      .getOrCreate()

    // SparkSession has implicits
    import spark.implicits._

    // book schema
    val bookSchema = StructType(
      StructField("ISBN", StringType, nullable = true) ::
        StructField("Title", StringType, nullable = true) ::
        StructField("Author", StringType, nullable = true) ::
        StructField("Year", IntegerType, nullable = true) ::
        StructField("Publisher", StringType, nullable = true) ::
        Nil
    )

    // rating schema
    val ratingSchema = StructType(
      StructField("USER-ID", IntegerType, nullable = true) ::
        StructField("ISBN", IntegerType, nullable = true) ::
        StructField("Rating", IntegerType, nullable = true) ::
        Nil
    )

    // read books
    val bookDf = spark.read.format("csv")
      .option("header", value = true)
      .option("delimiter", ";")
      .option("mode", "DROPMALFORMED")
      .schema(bookSchema)
      .load(RecommendationSystem.getClass.getResource("/dataset/Books.csv").getPath)
      .cache()
      .as("books")
    bookDf.printSchema()
    bookDf.show(10)

    // read ratings
    val ratingDf = spark.read.format("csv")
      .option("header", value = true)
      .option("delimiter", ";")
      .option("mode", "DROPMALFORMED")
      .schema(ratingSchema)
      .load(RecommendationSystem.getClass.getResource("/dataset/Ratings.csv").getPath)
      .cache()
      .as("ratings")
    ratingDf.printSchema()
    ratingDf.show(10)

    // join dfs
    val jdf = ratingDf.join(bookDf, $"ratings.ISBN" === $"books.ISBN")
      .select(
        $"ratings.USER-ID".as("userId"),
        $"ratings.Rating".as("rating"),
        $"ratings.ISBN".as("isbn"),
        $"books.Title".as("title"),
        $"books.Author".as("author"),
        $"books.Year".as("year"),
        $"books.Publisher".as("publisher")
      )
    jdf.printSchema()
    jdf.show(10)


    // Split the data into training and test sets (for example, 80% training and 20% test)
    val Array(trainingData, testData) = jdf.randomSplit(Array(0.8, 0.2))

    // build recommendation model with als algorithm
    val als = new ALS()
      .setMaxIter(15)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("isbn")
      .setRatingCol("rating")
    val alsModel = als.fit(trainingData)

    // evaluate the als model
    // compute root mean square error(rmse) with test data for evaluation
    // set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    alsModel.setColdStartStrategy("drop")
    val predictions = alsModel.transform(testData)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"root mean square error $rmse")


    val route =
      pathPrefix("recommendations") {
        concat(
          path("user" / IntNumber) { userId =>
            val userRecommendations = alsModel.recommendForUserSubset(Seq(userId).toDF("userId"), 10)
              .select($"userId", explode($"recommendations").as("rec"))
              .select($"userId", $"rec.ISBN".as("recommendedBook"))

            val recommendedBooks = userRecommendations
              .join(bookDf, userRecommendations("recommendedBook") === bookDf("ISBN"))
              .select($"userId", $"recommendedBook", $"Title", $"Author", $"Year", $"Publisher")
              .limit(10)
              .collect()

            val tableHeaders = Seq("User ID", "Recommended Book", "Title", "Author", "Year", "Publisher")

            val tableRows = recommendedBooks.map { row =>
              val rowData = Seq(
                row.getAs[Int]("userId").toString,
                row.getAs[Int]("recommendedBook").toString,
                row.getAs[String]("Title"),
                row.getAs[String]("Author"),
                row.getAs[Int]("Year").toString,
                row.getAs[String]("Publisher")
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
          },
          path("userchoice" / IntNumber) { userId =>
            val userRatings = jdf.filter($"userId" === userId)
            val userHighestRatedBooks = userRatings.sort($"rating".desc).limit(10)

            val tableHeaders = Seq("User ID", "Book ID", "Title", "Author", "Year", "Publisher")

            val tableRows = userHighestRatedBooks.collect().map { row =>
              val rowData = Seq(
                userId.toString,
                row.getAs[Int]("isbn").toString,
                row.getAs[String]("title"),
                row.getAs[String]("author"),
                row.getAs[Int]("year").toString,
                row.getAs[String]("publisher")
              )
              s"<tr>${rowData.map(data => s"<td>$data</td>").mkString}</tr>"
            }.mkString

            val htmlTable =
              s"""
                 |<html>
                 |<head><title>User's Highest Rated Books</title></head>
                 |<body>
                 |<h1>Top 10 Highly Rated Books for User $userId</h1>
                 |<table border="1">
                 |  <tr>${tableHeaders.map(header => s"<th>$header</th>").mkString}</tr>
                 |  $tableRows
                 |</table>
                 |</body>
                 |</html>
                 |""".stripMargin

            complete(HttpEntity(ContentTypes.`text/html(UTF-8)`, htmlTable))
          }
        )
      }


    val bindingFuture = Http().bindAndHandle(route, "localhost", 8080)

        println(s"Server online at http://localhost:8080/\nPress RETURN to stop...")
        scala.io.StdIn.readLine() // Let the server run until user presses return
        bindingFuture
          .flatMap(_.unbind())
          .onComplete(_ => system.terminate())

        // Stop the SparkSession
        spark.stop()

       }
  }





