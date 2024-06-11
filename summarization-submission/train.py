import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from tira.rest_api_client import Client
import joblib

def preprocess_text(text):
    return text.lower()

def extract_sentences(text):
    return text.split('. ')

def train_model(train_data):
    documents = [preprocess_text(doc['story']) for doc in train_data]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    return vectorizer, X

if __name__ == "__main__":
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")
    train_data = df.reset_index().to_dict(orient='records')

    vectorizer, X = train_model(train_data)

    model_dir = Path(__file__).parent
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    joblib.dump(X, model_dir / "tfidf_matrix.joblib")
