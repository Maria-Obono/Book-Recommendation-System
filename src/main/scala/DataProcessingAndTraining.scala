import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import RecommendationSystem._
trait DataProcessingAndTraining {
  def readAndJoinData(spark: SparkSession): (DataFrame, DataFrame) = {
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
        $"ratings.ISBN".as("ISBN"),
        $"books.Book-Title".as("title"),
        $"books.Book-Author".as("author"),
        $"books.Year-Of-Publication".as("year"),
        $"books.Publisher".as("publisher"),
        $"books.Image-URL-S".as("urlS"),
        $"books.Image-URL-M".as("urlM"),
        $"books.Image-URL-L".as("urlL")
      ).toDF()
    (bookDf, jdf)
  }


  def splitData(jdf: DataFrame): Array[DataFrame] = {
    jdf.randomSplit(Array(0.8, 0.2))
  }

  def trainALSModel(trainingData: DataFrame): ALSModel = {
    val als = new ALS()
      .setMaxIter(15)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("ISBN")
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
}
