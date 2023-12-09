# Book-Recommendation-System
## Summary
This Book Recommendation System provides personalized book recommendations based on user preferences and historical data. This project implements a Collaborative Filtering Recommendation System using the ALS(Alternating Least Square) algorithm from Spark MLlib.
## Table Of Contents
- [Tech Stack Used](#Tech-Stack-Used)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [API Endpoints](#api-endpoints)
    - [/recommendations/user/:userId](#user)
    - [/recommendations/userchoice/:userId](#recommendations)
- [Usage Examples](#usage-examples)
- [Sample Requests](#sample-requests)

### Tech Stack Used
- Scala 2.13.7
- Apache Spark , Core, SQL, MLIB 3.5.0
- Akka Http toolkit 10.6.0
### Getting Started
1. clone this repo
```bash
   git clone https://github.com/Maria-Obono/Book-Recommendation-System.git
```
2. Run Application in IDE
- To run the Spark Actor Recommendation application from your IDE, follow these steps:
Locate the BookRecommendationSystem Scala class in your IDE.
- Run the main method in the RecommendationSystem class.
- This will load data, process data and train ALS model.
- Once Model is trained a port ```8080``` is exposed to invoke API

### Dataset
- Dataset is available from kaggle at
https://www.kaggle.com/datasets/saurabhbagchi/books-dataset/
- This contains three csv files books.csv, ratings.csv, users.csv
- These files are added as part of project structure and available in git at
  ```src/scala/books_data/```

### Training the Model
- Alternating Least squares Algorithms is used.
- The data is cleaned and merged after which data is split into two parts ```(0.8,0.2)```
- 80 percent of data is trained to fit the model.

### Model Evaluation
- 20 percent of remaining split data is used to test the model and tune ALS.
- The model is evaluated for lower RMSE as possible 
- We achieved 5.5 RMSE


### API Endpoints
Two API endpoints were exposed on port: ```8080```.

### /recommendations/user/:userId
This Api will fetch top 10 recommendations for the given userId  
```GET http://localhost:8080/recommendations/user/:userId```
### /recommendations/userchoice/:userId
This Api will fetch top 10 Highly rated books for the given userId  
```GET http://localhost:8080/recommendations/userchoice/:userId```
### Sample Requests
APIs will fetch the recommended books as html table
- ```GET http://localhost:8088/recommendations/user/11944```
- ```GET http://localhost:8088/recommendations/userchoice/277378```