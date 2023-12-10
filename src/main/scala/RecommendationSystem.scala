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
        StructField("Book-Title", StringType, nullable = true) ::
        StructField("Book-Author", StringType, nullable = true) ::
        StructField("Year-Of-Publication", IntegerType, nullable = true) ::
        StructField("Publisher", StringType, nullable = true) ::
        StructField("Image-URL-S", StringType, nullable = true) ::
        StructField("Image-URL-M", StringType, nullable = true) ::
        StructField("Image-URL-L", StringType, nullable = true) ::

        Nil
    )

    // rating schema
    val ratingSchema = StructType(
      StructField("USER-ID", IntegerType, nullable = true) ::
        StructField("ISBN", IntegerType, nullable = true) ::
        StructField("Book-Rating", IntegerType, nullable = true) ::
        Nil
    )

    // read books
    val bookDf = spark.read.format("csv")
      .option("header", value = true)
      .option("delimiter", ";")
      .option("mode", "DROPMALFORMED")
      .schema(bookSchema)
      .load(RecommendationSystem.getClass.getResource("/books_data/Books.csv").getPath)
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
      .load(RecommendationSystem.getClass.getResource("/books_data/Ratings.csv").getPath)
      .cache()
      .as("ratings")
    ratingDf.printSchema()
    ratingDf.show(10)

    // join dfs
    val jdf = ratingDf.join(bookDf, $"ratings.ISBN" === $"books.ISBN")
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
              .select($"userId", $"recommendedBook", $"Book-Title", $"Book-Author", $"Year-Of-Publication", $"Publisher", $"Image-URL-S", $"Image-URL-M", $"Image-URL-L" )
              .limit(10)
              .collect()

            val tableHeaders = Seq("User ID", "Recommended Book", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-S", "Image-URL-M", "Image-URL-L")

            val tableRows = recommendedBooks.map { row =>
              val rowData = Seq(
                row.getAs[Int]("userId").toString,
                row.getAs[Int]("recommendedBook").toString,
                row.getAs[String]("Book-Title"),
                row.getAs[String]("Book-Author"),
                row.getAs[Int]("Year-Of-Publication").toString,
                row.getAs[String]("Publisher"),
                row.getAs[String]("Image-URL-S"),
                row.getAs[String]("Image-URL-S"),
                row.getAs[String]("Image-URL-S"),

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

            val tableHeaders = Seq("User ID", "Recommended Book", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-S", "Image-URL-M", "Image-URL-L")

            val tableRows = userHighestRatedBooks.collect().map { row =>
              val rowData = Seq(
                userId.toString,
                row.getAs[Int]("isbn").toString,
                row.getAs[String]("title"),
                row.getAs[String]("author"),
                row.getAs[Int]("year").toString,
                row.getAs[String]("publisher"),
                row.getAs[String]("Image-URL-S"),
                row.getAs[String]("Image-URL-S"),
                row.getAs[String]("Image-URL-S"),
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





