import json
from tira.rest_api_client import Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
import joblib  

def load_data(tira, dataset_name):
    text = tira.pd.inputs("nlpbuw-fsu-sose-24", dataset_name).set_index("id")
    labels = tira.pd.truths("nlpbuw-fsu-sose-24", dataset_name).set_index("id")
    df = text.join(labels)
    return df

if __name__ == "__main__":
    tira = Client()
    train_df = load_data(tira, "paraphrase-identification-train-20240515-training")

    train_df['combined'] = train_df['sentence1'] + ' ' + train_df['sentence2']
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_df['combined'])
    y_train = train_df['label']

    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'logistic_regression_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

    
