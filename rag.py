import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from google import genai

# Load environment variables
load_dotenv()

# -----------------------------
# 1. Initialize services
# -----------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index = pc.Index("company-policy-rag")

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# -----------------------------
# 2. User question
# -----------------------------

question = "How many days of annual leave do employees receive?"

# -----------------------------
# 3. Convert question to embedding
# -----------------------------

query_embedding = embedding_model.encode(question).tolist()

# -----------------------------
# 4. Retrieve relevant chunks
# -----------------------------

results = index.query(
    vector=query_embedding,
    top_k=3,
    include_metadata=True
)

# -----------------------------
# 5. Build RAG context
# -----------------------------

context = "\n\n".join(
    match["metadata"]["text"]
    for match in results["matches"]
)

# -----------------------------
# 6. Build prompt
# -----------------------------

prompt = f"""
You are a company policy assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context, say:
"I could not find this information in the company policy."

Context:
{context}

User question:
{question}
"""

# -----------------------------
# 7. Send context + question to Gemini
# -----------------------------

response = gemini_client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

# -----------------------------
# 8. Display result
# -----------------------------

print("\n===== QUESTION =====")
print(question)

print("\n===== ANSWER =====")
print(response.text)

