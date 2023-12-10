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

    // Import Spark implicits
    import spark.implicits._

    // Define book schema
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

    // Define rating schema
    val ratingSchema = StructType(
      StructField("USER-ID", IntegerType, nullable = true) ::
        StructField("ISBN", IntegerType, nullable = true) ::
        StructField("Book-Rating", IntegerType, nullable = true) ::
        Nil
    )

    // Read books data
    val bookDf = spark.read.format("csv")
      .option("header", value = true)
      .option("delimiter", ";")
      .option("mode", "DROPMALFORMED")
      .schema(bookSchema)
      .load(RecommendationSystem.getClass.getResource("/books_data/Books.csv").getPath)
      .cache()
      .as("books")

    // Display books schema and sample data
    bookDf.printSchema()
    bookDf.show(10)

    // Read ratings data
    val ratingDf = spark.read.format("csv")
      .option("header", value = true)
      .option("delimiter", ";")
      .option("mode", "DROPMALFORMED")
      .schema(ratingSchema)
      .load(RecommendationSystem.getClass.getResource("/books_data/Ratings.csv").getPath)
      .cache()
      .as("ratings")

    // Display ratings schema and sample data
    ratingDf.printSchema()
    ratingDf.show(10)

    // Join dataframes
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

    // Display joined dataframe schema and sample data
    jdf.printSchema()
    jdf.show(10)

    // Split data into training and test sets
    val Array(trainingData, testData) = jdf.randomSplit(Array(0.8, 0.2))

    // Build recommendation model with ALS algorithm
    val als = new ALS()
      .setMaxIter(15)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("isbn")
      .setRatingCol("rating")
    val alsModel = als.fit(trainingData)

    // Evaluate the ALS model
    alsModel.setColdStartStrategy("drop")
    val predictions = alsModel.transform(testData)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"root mean square error $rmse")

    // Define routes for HTTP server
    val route =
      pathPrefix("recommendations") {
        concat(
          // Endpoint to fetch recommendations for a user
          path("user" / IntNumber) { userId =>

            // Retrieve recommendations for a user using ALS model
            val userRecommendations = alsModel.recommendForUserSubset(Seq(userId).toDF("userId"), 10)
              .select($"userId", explode($"recommendations").as("rec"))
              .select($"userId", $"rec.ISBN".as("recommendedBook"))

            // Join recommended books with book data to retrieve details
            val recommendedBooks = userRecommendations
              .join(bookDf, userRecommendations("recommendedBook") === bookDf("ISBN"))
              .select($"userId", $"recommendedBook", $"Book-Title", $"Book-Author", $"Year-Of-Publication", $"Publisher", $"Image-URL-S", $"Image-URL-M", $"Image-URL-L" )
              .limit(10)
              .collect()

            // Define table headers for HTML output
            val tableHeaders = Seq("User ID", "Recommended Book", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-S", "Image-URL-M", "Image-URL-L")

            // Create table rows with book details in HTML format
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

            // HTML content to display recommended books in a table
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

            // Return HTML response with recommended books table
            complete(HttpEntity(ContentTypes.`text/html(UTF-8)`, htmlTable))
          },
          path("userchoice" / IntNumber) { userId =>

            // Retrieve a user's highest rated books
            val userRatings = jdf.filter($"userId" === userId)
            val userHighestRatedBooks = userRatings.sort($"rating".desc).limit(10)

            // Define table headers for HTML output
            val tableHeaders = Seq("User ID", "Recommended Book", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher", "Image-URL-S", "Image-URL-M", "Image-URL-L")

            // Create table rows with user's highest rated books in HTML format
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

            // HTML content to display user's highest rated books in a table
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
            // Return HTML response with user's highest rated books table
            complete(HttpEntity(ContentTypes.`text/html(UTF-8)`, htmlTable))
          }
        )
      }

    // Bind the defined routes to the specified host and port
    val bindingFuture = Http().bindAndHandle(route, "localhost", 8080)

    // Print a message indicating the server is online and how to stop it
        println(s"Server online at http://localhost:8080/\nPress RETURN to stop...")

    // Wait for user input to stop the server (keep it running until user input)
        scala.io.StdIn.readLine() // Let the server run until user presses return
    // Unbind the server to stop listening to incoming connections
        bindingFuture
          .flatMap(_.unbind())
          .onComplete(_ => system.terminate()) // Terminate the ActorSystem and associated resources

        // Stop the SparkSession
        spark.stop()

       }
  }





