# Movie Review Sentiment Analysis

This is a simple web application built with Flask that predicts the sentiment (positive or negative) of IMDb movie reviews using a pre-trained Multinomial Naive Bayes classifier.

## Overview

The application takes user input in the form of a movie review and predicts whether the sentiment of the review is positive or negative based on a pre-trained machine learning model. The model was trained on IMDb movie review data using a combination of text vectorization with CountVectorizer and classification with Multinomial Naive Bayes.

## Usage

To use the application, follow these steps:

1. Clone the repository:
   
   git clone https://github.com/Gaurav7506/review-sentiment-analysis.git
Navigate to the project directory:


2. Install dependencies (Flask, scikit-learn, etc.):
   ## pip install -r requirements.txt

3. Run the Flask application:
   ## python app.py

4. Open your web browser and go to http://127.0.0.1:5000 to access the application.

5. Enter a movie review in the input field and submit the form to see the predicted sentiment.

## Files and Directory Structure
app.py: The main Flask application file containing routes and model loading.

Naive_Bayes_model_imdb.pkl: Pickled Multinomial Naive Bayes model trained on IMDb movie review data.

countVect_imdb.pkl: Pickled CountVectorizer used for text vectorization during model training.

templates/: Directory containing HTML templates for rendering the web pages..

home.html: HTML template for the home page with the input form.

result.html: HTML template for displaying the prediction result.
