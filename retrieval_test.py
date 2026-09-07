import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index("company-policy-rag")

# User question
query = "How many days of annual leave do employees receive?"

# Convert question into embedding
query_embedding = model.encode(query).tolist()

# Search Pinecone
results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

print("\nQuery:")
print(query)

print("\nTop matching chunks:")

for match in results["matches"]:
    print("\n--------------------")
    print("Score:", match["score"])
    print("ID:", match["id"])
    print("Page:", match["metadata"]["page"])
    print("Text:")
    print(match["metadata"]["text"])