# ======================= Applied Naive Bayes =======================

# import pandas as pd
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier

# # Download required NLTK resources
# print("Downloading required NLTK resources...")
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# # Load the dataset
# dataset_1 = pd.read_csv('IMDB Dataset.csv')

# # Preprocessing function
# def preprocess_text(text):
#     lemmatizer = WordNetLemmatizer()
#     text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and digits
#     tokens = text.split()  # Tokenize
#     stop_words = set(stopwords.words('english'))
#     tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
#     return ' '.join(tokens)

# # Apply preprocessing to the 'review' column
# print("Preprocessing reviews...")
# dataset_1['preprocessed_review'] = dataset_1['review'].apply(preprocess_text)

# # Convert 'sentiment' to binary labels (positive: 1, negative: 0)
# dataset_1['label'] = dataset_1['sentiment'].map({'positive': 1, 'negative': 0})

# # Split the data into training and testing sets
# X = dataset_1['preprocessed_review']
# y = dataset_1['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Convert text data to Bag-of-Words features
# vectorizer = CountVectorizer(max_features=5000, min_df=2, max_df=0.95)
# X_train_bow = vectorizer.fit_transform(X_train)
# X_test_bow = vectorizer.transform(X_test)

# # Define classifiers with their parameter grids
# classifiers = {
#     'Naive Bayes': {
#         'model': MultinomialNB(),
#         'params': {
#             'alpha': [0.1, 0.5, 1.0]
#         }
#     },
#     # 'Logistic Regression': {
#     #     'model': LogisticRegression(random_state=42, max_iter=1000),
#     #     'params': {
#     #         'C': [0.1, 1.0, 10.0],
#     #         'solver': ['lbfgs', 'liblinear']
#     #     }
#     # },
#     # 'Random Forest': {
#     #     'model': RandomForestClassifier(random_state=42),
#     #     'params': {
#     #         'n_estimators': [100],
#     #         'max_depth': [10],
#     #         'min_samples_split': [2]
#     #     }
#     # }
# }

# # Dictionary to store results
# results = {}

# # Train and evaluate each classifier
# print("Training and evaluating classifiers...")
# for name, clf_info in classifiers.items():
#     print(f"\nTraining {name}...")
    
#     # Perform grid search
#     grid_search = GridSearchCV(
#         clf_info['model'],
#         clf_info['params'],
#         cv=5,
#         n_jobs=-1,
#         scoring='accuracy',
#         verbose=1
#     )
    
#     # Fit the model
#     grid_search.fit(X_train_bow, y_train)
    
#     # Make predictions
#     y_pred = grid_search.predict(X_test_bow)
    
#     # Store results
#     results[name] = {
#     'best_model': grid_search.best_estimator_,
#     'best_params': grid_search.best_params_,
#     'best_score': grid_search.best_score_,
#     'test_accuracy': accuracy_score(y_test, y_pred),
#     'classification_report': classification_report(y_test, y_pred)
# }


# # Print results
# print("\nClassification Results:")
# print("=" * 50)
# for name, result in results.items():
#     print(f"\n{name}:")
#     print(f"Best Parameters: {result['best_params']}")
#     print(f"Best Cross-validation Score: {result['best_score']:.4f}")
#     print(f"Test Accuracy: {result['test_accuracy']:.4f}")
#     print("\nClassification Report:")
#     print(result['classification_report'])
#     print("-" * 50)

# # Function to make predictions on new text
# def predict_sentiment(text, classifier_name='Naive Bayes'):
#     # Preprocess the text
#     processed_text = preprocess_text(text)
    
#     # Transform to BOW
#     text_bow = vectorizer.transform([processed_text])
    
#     # Get the best model for the specified classifier
#     best_model = results[classifier_name]['best_model']
    
#     # Make prediction
#     prediction = best_model.predict(text_bow)[0]
    
#     return "Positive" if prediction == 1 else "Negative"

# # Example predictions
# test_texts = [
#     "This movie was amazing and absolutely fantastic!",
#     "Worst film I've ever seen, completely disappointing.",
#     "An okay movie with some good moments."
# ]

# print("\nExample Predictions:")
# for text in test_texts:
#     print(f"Text: {text}")
#     print(f"Sentiment: {predict_sentiment(text)}\n")

# ======================= Applied Logistic Regresion =======================

# import pandas as pd
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier

# # Download required NLTK resources
# print("Downloading required NLTK resources...")
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# # Load the dataset
# dataset_1 = pd.read_csv('IMDB Dataset.csv')

# # Preprocessing function
# def preprocess_text(text):
#     lemmatizer = WordNetLemmatizer()
#     text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and digits
#     tokens = text.split()  # Tokenize
#     stop_words = set(stopwords.words('english'))
#     tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
#     return ' '.join(tokens)

# # Apply preprocessing to the 'review' column
# print("Preprocessing reviews...")
# dataset_1['preprocessed_review'] = dataset_1['review'].apply(preprocess_text)

# # Convert 'sentiment' to binary labels (positive: 1, negative: 0)
# dataset_1['label'] = dataset_1['sentiment'].map({'positive': 1, 'negative': 0})

# # Split the data into training and testing sets
# X = dataset_1['preprocessed_review']
# y = dataset_1['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Convert text data to Bag-of-Words features
# vectorizer = CountVectorizer(max_features=5000, min_df=2, max_df=0.95)
# X_train_bow = vectorizer.fit_transform(X_train)
# X_test_bow = vectorizer.transform(X_test)

# # Define classifiers with their parameter grids
# classifiers = {
#     # 'Naive Bayes': {
#     #     'model': MultinomialNB(),
#     #     'params': {
#     #         'alpha': [0.1, 0.5, 1.0]
#     #     }
#     # },
#     'Logistic Regression': {
#         'model': LogisticRegression(random_state=42, max_iter=1000),
#         'params': {
#             'C': [0.1, 1.0, 10.0],
#             'solver': ['lbfgs', 'liblinear']
#         }
#     },
#     # 'Random Forest': {
#     #     'model': RandomForestClassifier(random_state=42),
#     #     'params': {
#     #         'n_estimators': [100],
#     #         'max_depth': [10],
#     #         'min_samples_split': [2]
#     #     }
#     # }
# }

# # Dictionary to store results
# results = {}

# # Train and evaluate each classifier
# print("Training and evaluating classifiers...")
# for name, clf_info in classifiers.items():
#     print(f"\nTraining {name}...")
    
#     # Perform grid search
#     grid_search = GridSearchCV(
#         clf_info['model'],
#         clf_info['params'],
#         cv=5,
#         n_jobs=-1,
#         scoring='accuracy',
#         verbose=1
#     )
    
#     # Fit the model
#     grid_search.fit(X_train_bow, y_train)
    
#     # Make predictions
#     y_pred = grid_search.predict(X_test_bow)
    
#     # Store results
#     results[name] = {
#     'best_model': grid_search.best_estimator_,
#     'best_params': grid_search.best_params_,
#     'best_score': grid_search.best_score_,
#     'test_accuracy': accuracy_score(y_test, y_pred),
#     'classification_report': classification_report(y_test, y_pred)
# }


# # Print results
# print("\nClassification Results:")
# print("=" * 50)
# for name, result in results.items():
#     print(f"\n{name}:")
#     print(f"Best Parameters: {result['best_params']}")
#     print(f"Best Cross-validation Score: {result['best_score']:.4f}")
#     print(f"Test Accuracy: {result['test_accuracy']:.4f}")
#     print("\nClassification Report:")
#     print(result['classification_report'])
#     print("-" * 50)

# # Function to make predictions on new text
# def predict_sentiment(text, classifier_name='Logistic Regression'):
#     # Preprocess the text
#     processed_text = preprocess_text(text)
    
#     # Transform to BOW
#     text_bow = vectorizer.transform([processed_text])
    
#     # Get the best model for the specified classifier
#     best_model = results[classifier_name]['best_model']
    
#     # Make prediction
#     prediction = best_model.predict(text_bow)[0]
    
#     return "Positive" if prediction == 1 else "Negative"

# # Example predictions
# test_texts = [
#     "This movie was amazing and absolutely fantastic!",
#     "Worst film I've ever seen, completely disappointing.",
#     "An okay movie with some good moments."
# ]

# print("\nExample Predictions:")
# for text in test_texts:
#     print(f"Text: {text}")
#     print(f"Sentiment: {predict_sentiment(text)}\n")

# ======================= Applied Random Forest =======================

# import pandas as pd
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier

# # Download required NLTK resources
# print("Downloading required NLTK resources...")
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# # Load the dataset
# dataset_1 = pd.read_csv('IMDB Dataset.csv')

# # Preprocessing function
# def preprocess_text(text):
#     lemmatizer = WordNetLemmatizer()
#     text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and digits
#     tokens = text.split()  # Tokenize
#     stop_words = set(stopwords.words('english'))
#     tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
#     return ' '.join(tokens)

# # Apply preprocessing to the 'review' column
# print("Preprocessing reviews...")
# dataset_1['preprocessed_review'] = dataset_1['review'].apply(preprocess_text)

# # Convert 'sentiment' to binary labels (positive: 1, negative: 0)
# dataset_1['label'] = dataset_1['sentiment'].map({'positive': 1, 'negative': 0})

# # Split the data into training and testing sets
# X = dataset_1['preprocessed_review']
# y = dataset_1['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Convert text data to Bag-of-Words features
# vectorizer = CountVectorizer(max_features=5000, min_df=2, max_df=0.95)
# X_train_bow = vectorizer.fit_transform(X_train)
# X_test_bow = vectorizer.transform(X_test)

# # Define classifiers with their parameter grids
# classifiers = {
#     # 'Naive Bayes': {
#     #     'model': MultinomialNB(),
#     #     'params': {
#     #         'alpha': [0.1, 0.5, 1.0]
#     #     }
#     # },
#     # 'Logistic Regression': {
#     #     'model': LogisticRegression(random_state=42, max_iter=1000),
#     #     'params': {
#     #         'C': [0.1, 1.0, 10.0],
#     #         'solver': ['lbfgs', 'liblinear']
#     #     }
#     # },
#     'Random Forest': {
#         'model': RandomForestClassifier(random_state=42),
#         'params': {
#             'n_estimators': [100],
#             'max_depth': [10],
#             'min_samples_split': [2]
#         }
#     }
# }

# # Dictionary to store results
# results = {}

# # Train and evaluate each classifier
# print("Training and evaluating classifiers...")
# for name, clf_info in classifiers.items():
#     print(f"\nTraining {name}...")
    
#     # Perform grid search
#     grid_search = GridSearchCV(
#         clf_info['model'],
#         clf_info['params'],
#         cv=5,
#         n_jobs=-1,
#         scoring='accuracy',
#         verbose=1
#     )
    
#     # Fit the model
#     grid_search.fit(X_train_bow, y_train)
    
#     # Make predictions
#     y_pred = grid_search.predict(X_test_bow)
    
#     # Store results
#     results[name] = {
#     'best_model': grid_search.best_estimator_,
#     'best_params': grid_search.best_params_,
#     'best_score': grid_search.best_score_,
#     'test_accuracy': accuracy_score(y_test, y_pred),
#     'classification_report': classification_report(y_test, y_pred)
# }


# # Print results
# print("\nClassification Results:")
# print("=" * 50)
# for name, result in results.items():
#     print(f"\n{name}:")
#     print(f"Best Parameters: {result['best_params']}")
#     print(f"Best Cross-validation Score: {result['best_score']:.4f}")
#     print(f"Test Accuracy: {result['test_accuracy']:.4f}")
#     print("\nClassification Report:")
#     print(result['classification_report'])
#     print("-" * 50)

# # Function to make predictions on new text
# def predict_sentiment(text, classifier_name='Random Forest'):
#     # Preprocess the text
#     processed_text = preprocess_text(text)
    
#     # Transform to BOW
#     text_bow = vectorizer.transform([processed_text])
    
#     # Get the best model for the specified classifier
#     best_model = results[classifier_name]['best_model']
    
#     # Make prediction
#     prediction = best_model.predict(text_bow)[0]
    
#     return "Positive" if prediction == 1 else "Negative"

# # Example predictions
# test_texts = [
#     "This movie was amazing and absolutely fantastic!",
#     "Worst film I've ever seen, completely disappointing.",
#     "An okay movie with some good moments."
# ]

# print("\nExample Predictions:")
# for text in test_texts:
#     print(f"Text: {text}")
#     print(f"Sentiment: {predict_sentiment(text)}\n")

# ======================= Applied Deep learning using RNN =======================

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Download required NLTK resources
print("Downloading required NLTK resources...")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the dataset
dataset_1 = pd.read_csv('IMDB Dataset.csv')

# Preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and digits
    tokens = text.split()  # Tokenize
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the 'review' column
print("Preprocessing reviews...")
dataset_1['preprocessed_review'] = dataset_1['review'].apply(preprocess_text)

# Convert 'sentiment' to binary labels (positive: 1, negative: 0)
dataset_1['label'] = dataset_1['sentiment'].map({'positive': 1, 'negative': 0})

# Split the data into training and testing sets
X = dataset_1['preprocessed_review']
y = dataset_1['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Tokenization and Sequence Preparation for RNN
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure equal length
max_length = 200  # You can adjust this
X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# Prepare previous models (Bag of Words)
vectorizer = CountVectorizer(max_features=5000, min_df=2, max_df=0.95)
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# RNN Model
def create_rnn_model(input_dim, output_dim=1):
    model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=max_length),
        SimpleRNN(64, return_sequences=True),
        Dropout(0.5),
        SimpleRNN(32),
        Dense(16, activation='relu'),
        Dropout(0.5),
        Dense(output_dim, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Create and train RNN model
rnn_model = create_rnn_model(input_dim=5000)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)

# Train the model
print("Training RNN model...")
history = rnn_model.fit(
    X_train_padded, y_train, 
    epochs=10, 
    batch_size=32, 
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate RNN model
rnn_pred = (rnn_model.predict(X_test_padded) > 0.5).astype(int)
rnn_accuracy = accuracy_score(y_test, rnn_pred)
rnn_classification_report = classification_report(y_test, rnn_pred)

# Print RNN Results
print("\nRNN Classification Results:")
print("=" * 50)
print(f"Test Accuracy: {rnn_accuracy:.4f}")
print("\nClassification Report:")
print(rnn_classification_report)

# Function to make predictions on new text
def predict_sentiment_rnn(text):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Convert to sequence
    seq = tokenizer.texts_to_sequences([processed_text])
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    
    # Predict
    prediction = rnn_model.predict(padded_seq)[0][0]
    
    return "Positive" if prediction > 0.5 else "Negative"

# Example predictions
test_texts = [
    "This movie was amazing and absolutely fantastic!",
    "Worst film I've ever seen, completely disappointing.",
    "An okay movie with some good moments."
]

print("\nExample Predictions:")
for text in test_texts:
    print(f"Text: {text}")
    print(f"Sentiment: {predict_sentiment_rnn(text)}\n")

# ======================= Applied Deep learning using LSTM =======================

# import pandas as pd
# import numpy as np
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score

# # TensorFlow imports
# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

# # Download required NLTK resources
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# # Load the dataset
# dataset_1 = pd.read_csv('IMDB Dataset.csv')

# # Advanced Preprocessing Function
# def advanced_preprocess_text(text):
#     # More aggressive preprocessing
#     lemmatizer = WordNetLemmatizer()
    
#     # Remove HTML tags
#     text = re.sub(r'<.*?>', '', text)
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Remove special characters and numbers, keep apostrophes for contractions
#     text = re.sub(r'[^a-zA-Z\'\s]', '', text)
    
#     # Tokenize
#     tokens = text.split()
    
#     # Remove stop words and lemmatize
#     stop_words = set(stopwords.words('english'))
#     tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
#     return ' '.join(tokens)

# # Apply preprocessing
# print("Preprocessing reviews...")
# dataset_1['preprocessed_review'] = dataset_1['review'].apply(advanced_preprocess_text)

# # Convert sentiment to binary
# dataset_1['label'] = dataset_1['sentiment'].map({'positive': 1, 'negative': 0})

# # Split data
# X = dataset_1['preprocessed_review']
# y = dataset_1['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Tokenization
# tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
# tokenizer.fit_on_texts(X_train)

# # Convert to sequences
# X_train_seq = tokenizer.texts_to_sequences(X_train)
# X_test_seq = tokenizer.texts_to_sequences(X_test)

# # Pad sequences
# max_length = 250  # Increased from 200
# X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
# X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# # Improved Model Architecture
# def create_improved_model(input_dim, output_dim=1):
#     model = Sequential([
#         # Larger embedding dimension
#         Embedding(input_dim=10000, output_dim=256, input_length=max_length),
        
#         # LSTM instead of SimpleRNN for better sequence learning
#         LSTM(128, return_sequences=True),
#         Dropout(0.5),
        
#         LSTM(64),
#         Dropout(0.4),
        
#         # More dense layers for complex feature extraction
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(32, activation='relu'),
        
#         # Output layer
#         Dense(output_dim, activation='sigmoid')
#     ])
    
#     # Use Adam optimizer with lower learning rate
#     model.compile(optimizer=Adam(learning_rate=0.0005), 
#                   loss='binary_crossentropy', 
#                   metrics=['accuracy'])
    
#     return model

# # Create model
# model = create_improved_model(input_dim=10000)

# # Callbacks
# early_stopping = EarlyStopping(
#     monitor='val_accuracy',  # Changed from val_loss
#     patience=5, 
#     restore_best_weights=True
# )

# # Train with more sophisticated approach
# print("Training improved model...")
# history = model.fit(
#     X_train_padded, y_train, 
#     epochs=20,  # Increased epochs
#     batch_size=64,  # Larger batch size
#     validation_split=0.2,
#     callbacks=[early_stopping],
#     verbose=1
# )

# # Evaluate
# y_pred = (model.predict(X_test_padded) > 0.5).astype(int)
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print("\nImproved Model Results:")
# print(f"Accuracy: {accuracy:.4f}")
# print("\nClassification Report:")
# print(report)

# # Prediction function
# def predict_sentiment(text):
#     preprocessed = advanced_preprocess_text(text)
#     seq = tokenizer.texts_to_sequences([preprocessed])
#     padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
#     prediction = model.predict(padded)[0][0]
#     return "Positive" if prediction > 0.5 else "Negative"

# # Example predictions
# test_texts = [
#     "This movie was amazing and absolutely fantastic!",
#     "Worst film I've ever seen, completely disappointing.",
#     "An okay movie with some good moments."
# ]

# print("\nExample Predictions:")
# for text in test_texts:
#     print(f"Text: {text}")
#     print(f"Sentiment: {predict_sentiment(text)}\n")
