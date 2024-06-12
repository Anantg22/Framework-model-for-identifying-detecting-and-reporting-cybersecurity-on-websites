import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Read the text data
with open('firewall.txt', 'r') as file:
    data = file.readlines()

# Step 2: Preprocess the text
# For simplicity, let's just lowercase the text and tokenize it here
processed_data = [text.lower() for text in data]

# Reduce the size of the dataset
# For example, take the first 1000 samples
processed_data = [text.lower() for text in data[:500]]
y = np.random.randint(2, size=len(processed_data))

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(processed_data, y, test_size=0.2, random_state=42)

print("Training data shape:", len(X_train))
print("Testing data shape:", len(X_test))

# Optimize vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Limiting the number of features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("Shape of vectorized training data:", X_train_tfidf.shape)
print("Shape of vectorized testing data:", X_test_tfidf.shape)

# Step 4: Train a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=10, max_depth=10, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

# Step 5: Predict on the test set
predictions = rf_classifier.predict(X_test_tfidf)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)



# Combine training and testing data for prediction
combined_data = X_train + X_test

# Preprocess the combined data
processed_combined_data = [text.lower() for text in combined_data]

# Vectorize the combined data using the same TF-IDF vectorizer
X_combined_tfidf = tfidf_vectorizer.transform(processed_combined_data)

# Predict vulnerabilities using the trained classifier
predicted_labels_combined = rf_classifier.predict(X_combined_tfidf)

# Print the predicted labels for combined data
print("Predicted labels for combined data:")
for text, label in zip(combined_data, predicted_labels_combined):
    if label == 1:
        print(f"Vulnerability Detected: {text}")
    else:
        print(f"No Vulnerability Detected: {text}")
