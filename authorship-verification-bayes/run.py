from pathlib import Path
import json
from joblib import load
from sklearn.metrics import f1_score
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def load_data(dataset_name):
    tira = Client()
    data = tira.pd.inputs("nlpbuw-fsu-sose-24", dataset_name)
    return data

if __name__ == "__main__":
    # Load the test and trained data
    test_data = load_data("authorship-verification-validation-20240408-training")
    print("Test data:", test_data)
    model = load("model.joblib")
    predictions = model.predict(test_data["text"]) #making predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    with open(Path(output_directory) / "predictions.jsonl", "w", encoding="utf-8") as file: #prediction stored in jsonl file
        for sample_id, label in zip(test_data["id"], predictions):
            file.write(json.dumps({"id": sample_id, "generated": int(label)}) + "\n")


    # Calculating F1 score
    ground_truth = [int(sample_id) % 2 for sample_id in test_data["id"]]
    f1 = f1_score(ground_truth, predictions)

    # Save the F1 score
    with open(Path(output_directory) / "evaluation_scores.json", "w", encoding="utf-8") as file:
        json.dump({"F1 Score": f1}, file)
