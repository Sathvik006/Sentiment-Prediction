# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load the preprocessed dataset
data = pd.read_csv('Sentiment-Prediction/preprocessed_amazon_alexa_data.csv')
print(data.head())
data['processed_reviews'] = data['processed_reviews'].fillna('')

# Vectorize the text data
count_vect = CountVectorizer()
X = count_vect.fit_transform(data['processed_reviews'])
y = data['feedback']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate models
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    # Predict on test data
    y_pred = model.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_mat = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_mat)
    print("Classification Report:")
    print(report)
    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt="d",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

# Identify the best model based on accuracy
accuracies = {model_name: model.score(X_test, y_test) for model_name, model in models.items()}
best_model_name = max(accuracies, key=accuracies.get)
print(f"The best model is {best_model_name} with an accuracy of {accuracies[best_model_name]:.2f}")

# Save the best modelw
best_model = models[best_model_name]
joblib.dump(best_model, 'best_model.pkl')

# Save the vectorizer
joblib.dump(count_vect, 'vectorizer.pkl')
