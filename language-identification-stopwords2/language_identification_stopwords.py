import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from pathlib import Path

if __name__ == "__main__":
    # Initialize TIRA client
    tira = Client()

    # Load validation data (replaced by test data when run on TIRA)
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training")
    targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training")

    # Prepare the text and labels
    texts = text_validation['text']
    true_labels = targets_validation['lang']

    # Train a model on character n-grams
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 5))
    classifier = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')

    # Create a pipeline
    model = make_pipeline(vectorizer, classifier)

    # Train the model
    model.fit(texts, true_labels)

    # Predict the languages
    predicted_labels = model.predict(texts)

    # Calculate the F1 score
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"F1 Score: {f1:.4f}")

    # Prepare the prediction DataFrame
    prediction = pd.DataFrame({
        'id': text_validation['id'],
        'lang': predicted_labels
    })

    # Save the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
