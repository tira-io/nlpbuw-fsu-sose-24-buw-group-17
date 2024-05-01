from pathlib import Path
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client

def load_data(dataset_name):
    tira = Client()
    text = tira.pd.inputs("nlpbuw-fsu-sose-24", dataset_name)
    targets = tira.pd.truths("nlpbuw-fsu-sose-24", dataset_name)
    return text, targets

if __name__ == "__main__":
    #loading the data for training
    train_text, train_targets = load_data("authorship-verification-train-20240408-training")
    X_train = train_text["text"]
    y_train = train_targets["generated"] # text and labels are separated for the training sets in X_train and y_train
     
    # using logistic Regression for binary classification.
    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", LogisticRegression(max_iter=10000))  # Increase max_iter if needed
    ])
    model.fit(X_train, y_train)

    # Save the model
    dump(model, "/workspaces/nlpbuw-fsu-sose-24-buw-group-17/authorship-verification-bayes/model.joblib")

