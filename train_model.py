# train_model.py
import pandas as pd
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK tools (only once)
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
real = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")
real['label'] = 1
fake['label'] = 0

# Combine
df = pd.concat([real, fake], ignore_index=True).sample(frac=1, random_state=42)

# Preprocess text
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    words = text.lower().split()
    return ' '.join(lemmatizer.lemmatize(w) for w in words if w not in stop_words)

df['content'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(clean)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['content'])
y = df['label']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model training complete.")
print("📦 Model and vectorizer saved successfully.")
