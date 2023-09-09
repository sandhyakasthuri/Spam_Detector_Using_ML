# Spam_Detector_Using_ML
Description:
This is a Python script for detecting spam and non-spam (ham) emails using a Multinomial Naive Bayes classifier. The script uses a dataset of labeled emails to train the model and allows users to classify custom input text as spam or ham.

Table of Contents:

Installation
Usage
Customization
Contributing
License
Installation
Clone the repository to your local machine:

Copy code
git clone https://github.com/yourusername/email-spam-detector.git
Change to the project directory:

Copy code
cd email-spam-detector
Install the required dependencies. You can create a virtual environment for this project to manage dependencies separately.

Run the script:

Copy code
python main.py
You will be prompted to enter a text message for classification. Type in the text and press Enter. The script will predict whether the text is spam or ham and provide probabilities and classification metrics.

To exit the script, type 'exit' and press Enter.

Customization
Code Components Explanation
Loading Data: The script loads the email dataset from a CSV file, assuming it has column labels and text.

Text Preprocessing: Text data is preprocessed to make it suitable for modelling. This includes converting text to lowercase and removing punctuation.

Splitting Data: The dataset is split into training and testing sets to evaluate model performance.

Feature Extraction: The CountVectorizer from scikit-learn is used to convert text data into numerical features based on word counts. This creates a matrix where each row corresponds to an email, and each column represents a unique word.

Model Training: A Multinomial Naive Bayes classifier is trained on the transformed text data to classify emails as spam or ham.

Saving and Loading Model: The trained classifier and vectorizer are saved to files for future use and loaded when needed.

Custom Input Testing: Users can enter their own text for classification. The script preprocesses the input, transforms it using the loaded vectorizer, and predicts its label (spam or ham). It also calculates classification metrics.

Exit and Loop: Users can exit the script by typing 'exit'.

Customization Options
You can customize the preprocessing steps, feature extraction method, and machine learning model used for classification in the main.py script.

If you have your own dataset, replace spam_ham_dataset.csv with your data file, ensuring it has columns named label and text (or adjust the code accordingly).

let's see bit more about preprocessing and how we shaped the data so that it can be suitable as input to the ML model.
Lowercasing: All text is converted to lowercase. This ensures that words in different cases (e.g., "Hello" and "hello") are treated as the same word.

Punctuation Removal: Punctuation marks (such as commas, periods, and exclamation points) are removed from the text. This helps reduce the dimensionality of the feature space and ensures that punctuation doesn't affect classification.

Feature Extraction
Feature extraction is the process of converting text data into numerical features that can be used by machine learning algorithms. In this code, the primary feature extraction technique used is Count Vectorization:

Count Vectorization: This technique converts each email's text into a vector of word counts. It creates a matrix where each row corresponds to an email, and each column represents a unique word from the entire dataset. The value in each cell of the matrix represents the count of a specific word in the corresponding email. Count vectorization helps the machine learning model understand the frequency of words in each email.
Model Used
Multinomial Naive Bayes Classifier
The machine learning model used in this code is the Multinomial Naive Bayes classifier. Here's an overview of the model:

Naive Bayes: It's a probabilistic machine learning algorithm based on Bayes' theorem. The "naive" part of the name comes from the assumption that features (words in this case) are conditionally independent, which simplifies the probability calculations.

Multinomial Naive Bayes: This variant of the Naive Bayes classifier is commonly used for text classification tasks, including spam detection. It's well-suited for discrete data, such as word counts in text documents.

The model is trained on the transformed text data (word counts from the count vectorization) and learns to classify emails as either spam or ham based on the patterns it identifies in the training data.

Input Required for the Model
To use this code, you can input text data for classification. Here's how you can provide input and what the model expects:

Custom Input Text: Users are prompted to enter text for classification when running the script. They simply type in the text they want to classify as spam or ham and press Enter.

Input Data Format: The input text should be a single string, such as an email message or a sentence. The script then preprocesses this text and transforms it into the same format used during training (word counts).

Data Source
The dataset used in this code (spam_ham_dataset.csv) is not provided with the code but should be obtained separately. You can find similar email datasets on websites like Kaggle or UCI Machine Learning Repository. The dataset should have two columns: label (indicating spam or ham) and text (containing the email text). You can replace the file with your own dataset, ensuring it has the same format.

The specific dataset used to train the model is essential, as the quality and diversity of the data significantly impact the model's performance. It's recommended to use a well-labeled and representative dataset for training.

Feel free to customize and expand on these explanations in your README to provide users with a clear understanding of how the code works and how they can use it with their own data.




