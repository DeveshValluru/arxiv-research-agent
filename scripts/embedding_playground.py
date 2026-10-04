from sentence_transformers import SentenceTransformer


model = SentenceTransformer("BAAI/bge-small-en-v1.5")

sentences = [
    "LLM judges prefer the first answer they are shown.",
    "Language-model evaluators are biased toward whichever response comes first.",
    "Position bias affects pairwise comparisons made by AI graders.",
    "The judge sentenced the defendant to five years in prison.",
]


vectors = model.encode(sentences, normalize_embeddings=True)

print("shape:", vectors.shape)

similarity = vectors @ vectors.T

for i,sentence in enumerate(sentences):
    scores = " ".join(f"{similarity[i][j]:.2f}" for j in range(len(sentences)))
    print(scores, "|", sentence[:55])

