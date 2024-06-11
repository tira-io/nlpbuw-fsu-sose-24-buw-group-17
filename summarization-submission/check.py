import numpy as np
from pathlib import Path
print("Loading file...")
tfidf_matrix = np.load(Path(__file__).parent / "tfidf_matrix.npy")
print("Loaded array shape:", tfidf_matrix.shape)
