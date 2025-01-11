import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Download required NLTK resources
print("Downloading required NLTK resources...")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

review = ''
category = ''
dataName = ''
pos = ''
neg = ''

# true if chosen dataset is European Restaurant Reivews
europe = False
if (europe): 
    dataName = 'European Restaurant Reviews.csv'
    review = 'Review'
    category = 'Sentiment'
    pos = 'Positive'
    neg = 'Negative'

# true if chosen dataset is Spam text messages
spam = True
if (spam): 
    dataName = 'SPAM text message 20170820 - Data.csv'
    review = 'Message'
    category = 'Category'
    pos = 'spam'
    neg = 'ham'

# true if chosen dataset is IMDB 
IMDB = False
if (IMDB):
    dataName = 'IMDB Dataset.csv'
    review = 'review'
    category = 'sentiment'
    pos = 'positive'
    neg = 'negative'

# Load the dataset
dataset_1 = pd.read_csv(dataName)

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
dataset_1['preprocessed_review'] = dataset_1[review].apply(preprocess_text)

# Convert 'sentiment' to binary labels (positive: 1, negative: 0)
dataset_1['label'] = dataset_1[category].map({pos: 1, neg: 0})

# Split the data into training and testing sets
X = dataset_1['preprocessed_review']
y = dataset_1['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert text data to Bag-of-Words features
vectorizer = CountVectorizer(max_features=5000, min_df=2, max_df=0.95)
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Define classifiers with their parameter grids
classifiers = {
    'Naive Bayes': {
        'model': MultinomialNB(),
        'params': {
            'alpha': [0.1, 0.5, 1.0]
        }
    },
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, max_iter=1000),
        'params': {
            'C': [0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'liblinear']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [10, 50, 100],
            'max_depth': [5, 10, 25],
            'min_samples_split': [2, 5, 20]
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
    }
}


# Dictionary to store results
results = {}

# Train and evaluate each classifier
print("Training and evaluating classifiers...")
for name, clf_info in classifiers.items():
    print(f"\nTraining {name}...")
    
    # Perform grid search
    grid_search = GridSearchCV(
        clf_info['model'],
        clf_info['params'],
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train_bow, y_train)
    
    # Make predictions
    y_pred = grid_search.predict(X_test_bow)
    
    # Store results
    results[name] = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'test_accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

# Print results
print("\nClassification Results for {}:".format(dataName))
print("=" * 50)
for name, result in results.items():
    print(f"\n{name}:")
    print(f"Best Parameters: {result['best_params']}")
    print(f"Best Cross-validation Score: {result['best_score']:.4f}")
    print(f"Test Accuracy: {result['test_accuracy']:.4f}")
    print("\nClassification Report:")
    print(result['classification_report'])
    print("-" * 50)

# Function to make predictions on new text
def predict_sentiment(text, classifier_name='Logistic Regression'):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Transform to BOW
    text_bow = vectorizer.transform([processed_text])
    
    # Get the best model for the specified classifier
    best_model = results[classifier_name]['best_model']
    
    # Make prediction
    prediction = best_model.predict(text_bow)[0]
    
    return "Positive" if prediction == 1 else "Negative"
