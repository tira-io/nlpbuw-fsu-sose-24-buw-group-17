from pathlib import Path
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from tira.rest_api_client import Client
from tqdm import tqdm

def load_data(dataset_name):
    tira = Client()
    text = tira.pd.inputs("nlpbuw-fsu-sose-24", dataset_name)
    targets = tira.pd.truths("nlpbuw-fsu-sose-24", dataset_name)
    return text, targets

if __name__ == "__main__":
    print("Loading data...")
    train_text, train_targets = load_data("language-identification-train-20240429-training")
    X_train = train_text["text"]
    y_train = train_targets["lang"]

    model = make_pipeline(
        TfidfVectorizer(analyzer='char', ngram_range=(1, 3), sublinear_tf=True),
        LogisticRegression(max_iter=500, multi_class='auto', solver='liblinear')
    )

    with tqdm(total=len(X_train), desc="Training model") as pbar:
        model.fit(X_train, y_train)
        pbar.update(1)

        

    dump(model, Path(__file__).parent / "model.joblib")
