import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from tqdm import tqdm
import joblib

def preprocess_text(text):
    return text.lower()

def extract_sentences(text):
    return text.split('. ')

def summarize(vectorizer, tfidf_matrix, story):
    sentences = extract_sentences(story)
    sentences = [preprocess_text(sentence) for sentence in sentences]
    story_vector = vectorizer.transform(sentences)
    scores = cosine_similarity(story_vector, tfidf_matrix)
    ranked_sentences = [sentences[i] for i in np.argsort(scores.sum(axis=1))[-2:][::-1]]
    return '. '.join(ranked_sentences)

if __name__ == "__main__":
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")
    test_data = df.reset_index().to_dict(orient='records')

    vectorizer = joblib.load(Path(__file__).parent / "vectorizer.joblib")
    tfidf_matrix = joblib.load(Path(__file__).parent / "tfidf_matrix.joblib")

    predictions = []
    for article in tqdm(test_data):
        summary = summarize(vectorizer, tfidf_matrix, article['story'])
        predictions.append({
            "id": article['id'],
            "summary": summary
        })
    output_directory = get_output_directory(str(Path(__file__).parent))
    with open(Path(output_directory) / "predictions.jsonl", 'w') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")
