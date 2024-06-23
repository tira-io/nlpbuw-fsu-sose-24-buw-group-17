import joblib
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def extract_features(sentence):
    return [{ 
        'word': word,
        'is_first': i == 0,
        'is_last': i == len(sentence) - 1,
        'is_capitalized': word[0].upper() == word[0],
        'is_all_caps': word.upper() == word,
        'is_all_lower': word.lower() == word,
        'prefix-1': word[0],
        'prefix-2': word[:2],
        'prefix-3': word[:3],
        'suffix-1': word[-1],
        'suffix-2': word[-2:],
        'suffix-3': word[-3:],
        'prev_word': '' if i == 0 else sentence[i - 1],
        'next_word': '' if i == len(sentence) - 1 else sentence[i + 1],
    } for i, word in enumerate(sentence)]

if __name__ == "__main__":
    tira = Client()
    text_data = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training").to_dict(orient="records")

    sentences = [entry['sentence'].split() for entry in text_data]
    parent_dir = Path(__file__).resolve().parent
    model_path = parent_dir / 'crf_model.joblib'
    crf = joblib.load(model_path)
    X_val = [extract_features(sentence) for sentence in sentences]
    print("Generating predictions...")
    y_pred = [crf.predict_single(x) for x in tqdm(X_val, desc="Predicting")]
    pred_df = pd.DataFrame(text_data)
    pred_df['tags'] = y_pred
    output_directory = get_output_directory(str(Path(__file__).parent))
    pred_df = pred_df[['id', 'tags']]
    pred_df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
