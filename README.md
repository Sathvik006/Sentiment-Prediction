# Sentiment Prediction

## Project Description

**Problem Statement:** Create a classification model to predict sentiment (1 or 0) based on Amazon Alexa reviews.

**Context:** This dataset consists of nearly 3000 Amazon customer reviews, star ratings, date of review, variant, and feedback for various Amazon Alexa products (e.g., Alexa Echo, Echo dots, Alexa Firesticks, etc.) for sentiment analysis.

## Dataset

[Amazon Alexa Reviews Dataset](https://drive.google.com/file/d/1Sokf6jPQq7IAb92V0JO_nduwBJ3oZpgm/view?usp=share_link)

**Details of Features:**
1. `rating`: Product rating
2. `date`: Date on which the product was rated
3. `variation`: Variation of the product
4. `verified_reviews`: Verified reviews for Alexa
5. `feedback`: 1 (Positive) or 0 (Negative)

## Project Steps

1. Read the dataset
2. Handle null values (if any)
3. Preprocess the Amazon Alexa reviews:
   - Tokenizing words
   - Convert words to lower case
   - Remove punctuations
   - Remove stop words
   - Apply stemming
4. Transform the words into vectors using Count Vectorizer
5. Split data into training and test sets
6. Apply the following models on the training dataset and generate the predicted value for the test dataset:
   - Multinomial Naïve Bayes Classification
   - Logistic Regression
   - K-Nearest Neighbors (KNN) Classification
7. Predict the feedback for test data
8. Compute confusion matrix and classification report for each model
9. Report the model with the best accuracy

## Files

1. `preprocessing.py` - Contains the code for preprocessing the reviews.
2. `modeling.py` - Contains the code for training and evaluating the prediction models.

## Usage

1. Run `preprocessing.py` to preprocess the data and save it.
2. Run `modeling.py` to train and evaluate the models, and to identify and save the best model.

## Results

The results of the models are printed during the execution of `modeling.py`, including accuracy, confusion matrix, and classification report for each model. The best model is saved as `best_model.pkl` and the vectorizer as `vectorizer.pkl`.
