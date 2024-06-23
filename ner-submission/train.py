from tira.rest_api_client import Client
from sklearn_crfsuite import CRF
import joblib
from pathlib import Path
from tqdm import tqdm
from seqeval.metrics import classification_report

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

def preprocess_data(text_data, label_data):
    sentences = [entry['sentence'].split() for entry in text_data]
    tags = [entry['tags'] for entry in label_data]
    aligned_sentences, aligned_tags = [], []
    for sentence, tag in zip(sentences, tags):
        if len(sentence) == len(tag):
            aligned_sentences.append(sentence)
            aligned_tags.append(tag)
    
    return aligned_sentences, aligned_tags

def main():
    tira = Client()
    text_data = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-training-20240612-training").to_dict(orient="records")
    label_data = tira.pd.truths("nlpbuw-fsu-sose-24", "ner-training-20240612-training").to_dict(orient="records")

    sentences, tags = preprocess_data(text_data, label_data)
    X = [extract_features(sentence) for sentence in sentences]
    y = tags
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=False)
    print("Training the model...")
    crf.fit(X, y)
    y_pred = crf.predict(X)
    print("\nTraining Results:")
    print(classification_report(y, y_pred))

    parent_dir = Path(__file__).resolve().parent
    model_path = parent_dir / 'crf_model.joblib'
    joblib.dump(crf, model_path)

if __name__ == "__main__":
    main()
