from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Test sentences
texts = [
    "Employees receive 20 days of annual leave.",
    "How many vacation days do employees get?",
    "The company provides laptops to employees."
]


# Convert text into vectors
embeddings = model.encode(texts)


# Calculate cosine similarity
similarity_matrix = cosine_similarity(embeddings)


print("Number of texts:", len(texts))
print("Vector dimension:", len(embeddings[0]))

print("\nSimilarity Matrix:")
print(similarity_matrix)

print("\nSentence 1 vs Sentence 2:")
print(similarity_matrix[0][1])

print("\nSentence 1 vs Sentence 3:")
print(similarity_matrix[0][2])