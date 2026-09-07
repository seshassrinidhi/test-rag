from sentence_transformers import SentenceTransformer
from ingest import documents
# -----------------------------------
# 1. other embedding model and why we need vs embedding model vs Vector DB vs llm vs api key vs transformer
# -----------------------------------

# -----------------------------------
# 1. Load embedding model
# -----------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")
#test

# -----------------------------------
# 2. Extract text from our documents
# -----------------------------------

texts = [document["text"] for document in documents]


# -----------------------------------
# 3. Generate embeddings
# -----------------------------------

embeddings = model.encode(texts)


# -----------------------------------
# 4. Display results
# -----------------------------------

print("Total documents/chunks:", len(documents))
print("Total embeddings:", len(embeddings))
print("Vector dimension:", len(embeddings[0]))


# -----------------------------------
# 5. Show first chunk
# -----------------------------------

print("\nFirst chunk:")
print(documents[0]["text"])


# -----------------------------------
# 6. Show first chunk metadata
# -----------------------------------

print("\nFirst chunk metadata:")
print(documents[0]["metadata"])


# -----------------------------------
# 7. Show first embedding
# -----------------------------------

print("\nFirst embedding:")
print(embeddings[0])