from tira.rest_api_client import Client
import pandas as pd
from model import ParaphraseModel

if __name__ == "__main__":
    tira=Client()
    text=tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    labels=tira.pd.truths(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    
    model = ParaphraseModel()
    emb1=model.compute_embeddings(text['sentence1'].tolist())
    emb2=model.compute_embeddings(text['sentence2'].tolist())  
    sim=model.compute_cosine_similarities(emb1, emb2)
    df=text.join(labels)
    best_threshold = model.find_best_threshold(df, sim)
    print("Best threshold: {best_threshold}")
