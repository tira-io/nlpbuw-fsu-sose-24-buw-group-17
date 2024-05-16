from pathlib import Path
from joblib import load
import pandas as pd
from sklearn.metrics import f1_score
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from tqdm import tqdm

if __name__ == "__main__":
    tira = Client()
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training")
    targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training")

    texts = text_validation['text']
    true_labels = targets_validation['lang']

    model = load(Path(__file__).parent / "model.joblib")

    with tqdm(total=len(texts), desc="Predicting languages") as pbar:
        predicted_labels = []
        for text in texts:
            predicted_label = model.predict([text])[0]
            predicted_labels.append(predicted_label)
            pbar.update(1)

    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f"F1 Score: {f1:.4f}")

    prediction = pd.DataFrame({'id': text_validation['id'], 'lang': predicted_labels})
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
