from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import joblib  

def load_data(tira, dataset_name):
    text = tira.pd.inputs("nlpbuw-fsu-sose-24", dataset_name).set_index("id")
    return text

if __name__ == "__main__":
    
    tira = Client()
    model = joblib.load(Path(__file__).parent / "logistic_regression_model.pkl")
    vectorizer = joblib.load(Path(__file__).parent / "tfidf_vectorizer.pkl")

    test_df = load_data(tira, "paraphrase-identification-validation-20240515-training")
    test_df['combined'] = test_df['sentence1'] + ' ' + test_df['sentence2']
    X_test = vectorizer.transform(test_df['combined'])
    test_df['label'] = model.predict(X_test)

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions = test_df[['label']].reset_index()
    predictions.to_json(Path(output_directory) / "predictions.jsonl", orient='records', lines=True)

    print("Predictions saved.")
