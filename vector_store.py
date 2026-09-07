import os
from dotenv import load_dotenv
from pinecone import Pinecone

from embeddings import embeddings
from ingest import documents


# -----------------------------------
# 1. Load environment variables
# -----------------------------------

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")


# -----------------------------------
# 2. Connect to Pinecone
# -----------------------------------

pc = Pinecone(api_key=api_key)

index_name = "company-policy-rag"

index = pc.Index(index_name)


# -----------------------------------
# 3. Prepare vectors
# -----------------------------------

vectors = []

for i, (document, embedding) in enumerate(zip(documents, embeddings)):

    vector = {
        "id": f"policy_{i}",
        "values": embedding.tolist(),
        "metadata": {
            "text": document["text"],
            **document["metadata"]
        }
    }

    vectors.append(vector)


# -----------------------------------
# 4. Upload vectors
# -----------------------------------

index.upsert(vectors=vectors)

print(f"Uploaded {len(vectors)} vectors successfully!")