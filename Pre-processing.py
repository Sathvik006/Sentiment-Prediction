# Import necessary libraries
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string


data = pd.read_csv('Sentiment-Prediction/amazon_alexa_data.csv')

print(data.head())

print(data.isnull().sum())
data.dropna(inplace=True)

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.lower() not in stop_words]
    words = [word for word in words if word not in string.punctuation]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

data['processed_reviews'] = data['verified_reviews'].apply(preprocess_text)
print(data['processed_reviews'].head())

data.to_csv('Sentiment-Prediction/preprocessed_amazon_alexa_data.csv', index=False)
