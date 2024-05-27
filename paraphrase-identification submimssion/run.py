from pathlib import Path
from model import ParaphraseModel
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import numpy as np

def read_best_threshold(file_path):
    try:
        with open(file_path, 'r') as file:
            best_threshold=float(file.read().strip())
        return best_threshold
    except (FileNotFoundError, ValueError) as e:
        raise ValueError(f"Error reading the best threshold: {e}")

if __name__ == "__main__":
    tira=Client()
    df=tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")
    
    model=ParaphraseModel()
    emb1=model.compute_embeddings(df['sentence1'].tolist())
    emb2=model.compute_embeddings(df['sentence2'].tolist())
    sim=model.compute_cosine_similarities(emb1, emb2)
    sim=np.array(sim)
    best_threshold_path=Path(__file__).parent / "best_threshold.txt"
    best_threshold=read_best_threshold(best_threshold_path)
    df['label']=(sim>=best_threshold).astype(int)
    df = df.drop(columns=['sentence1', 'sentence2']).reset_index()
    
    
    output_directory=get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
