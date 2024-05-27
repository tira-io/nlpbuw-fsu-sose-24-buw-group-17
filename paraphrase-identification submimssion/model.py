from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

class ParaphraseModel:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def compute_embeddings(self, sentences):
        return self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=True)

    def compute_cosine_similarities(self, embeddings1, embeddings2):
        cosine_similarities = util.pytorch_cos_sim(embeddings1, embeddings2).cpu().numpy()
        return [cosine_similarities[i][i] for i in range(len(cosine_similarities))]

    def find_best_threshold(self, df, similarities):
        mccs={}
        for threshold in tqdm(sorted(set(similarities)), desc="Calculating MCC for different thresholds"):
            df['similarity'] = similarities
            tp=df[(df['similarity'] >= threshold) & (df['label'] == 1)].shape[0]
            fp=df[(df['similarity'] >= threshold) & (df['label'] == 0)].shape[0]
            tn=df[(df['similarity'] < threshold) & (df['label'] == 0)].shape[0]
            fn=df[(df['similarity'] < threshold) & (df['label'] == 1)].shape[0]
            try:
                mcc=(tp*tn-fp*fn) / (
                    (tp+fp)*(tp+fn)*(tn+fp)*(tn + fn)
                )**0.5
            except ZeroDivisionError:
                mcc=0
            mccs[threshold]=mcc
        best_threshold=max(mccs, key=mccs.get)
        return best_threshold
