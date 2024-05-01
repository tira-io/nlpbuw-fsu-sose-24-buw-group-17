from pathlib import Path
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def load_data(dataset_name, dataset_type):
    tira = Client()
    text = tira.pd.inputs("nlpbuw-fsu-sose-24", dataset_name)
    targets = tira.pd.truths("nlpbuw-fsu-sose-24", dataset_name)
    return text, targets

def calculate_metrics(predictions, ground_truth):
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    return precision, recall, f1

if __name__ == "__main__":
    # Load train and validation data
    train_text, train_targets = load_data("authorship-verification-train-20240408-training", "train")
    validation_text, validation_targets = load_data("authorship-verification-validation-20240408-training", "validation")


