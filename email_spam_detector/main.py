import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
data = pd.read_csv("spam_ham_dataset.csv")

# Split the dataset into features (X) and labels (y)
X = data['text']
y = data['label_num']

# Extract features using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X, y)

# Save the trained model and vectorizer for future use
joblib.dump(classifier, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "count_vectorizer.pkl")

# Load the trained model and vectorizer
classifier = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("count_vectorizer.pkl")

# Test the trained model with custom input
while True:
    custom_text = input("Enter a text to classify (or 'exit' to quit): ")
    if custom_text.lower() == 'exit':
        break

    # Preprocess the input text (similar to the training data preprocessing)
    custom_text = custom_text.lower()

    # Transform the input text using the loaded vectorizer
    custom_text_vectorized = vectorizer.transform([custom_text])

    # Predict the class (spam or ham)
    predicted_label = classifier.predict(custom_text_vectorized)

    if predicted_label[0] == 1:
        print("Predicted Label: Spam")
    else:
        print("Predicted Label: Ham")

    # Provide classification metrics for the custom input
    predicted_proba = classifier.predict_proba(custom_text_vectorized)
    spam_probability = predicted_proba[0, 1]
    ham_probability = predicted_proba[0, 0]

    print(f"Spam Probability: {spam_probability:.2f}")
    print(f"Ham Probability: {ham_probability:.2f}")

    # Calculate accuracy on the custom input
    y_true = [1] if custom_text.lower() == "spam" else [0]
    y_pred = predicted_label
    accuracy = accuracy_score(y_true, y_pred)

