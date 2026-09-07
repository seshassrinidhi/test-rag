from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------------
# 1. why we need langchain and what langchain_text_splitters


# -----------------------------------
# -----------------------------------
# 2. PDF information
# -----------------------------------

pdf_path = "Documents/company_policy.pdf"

document_id = "policy_001"
document_name = "company_policy.pdf"
department = "HR"
version = "2026"


# -----------------------------------
# 2. Read PDF page by page
# -----------------------------------

reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))


# -----------------------------------
# 3. Create the text splitter
# -----------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)


# -----------------------------------
# 4. Create chunks with metadata
# -----------------------------------

documents = []

for page_number, page in enumerate(reader.pages, start=1):

    page_text = page.extract_text()

    if not page_text:
        continue

    chunks = splitter.split_text(page_text)

    for chunk_number, chunk in enumerate(chunks, start=1):

        document = {
            "text": chunk,
            "metadata": {
                "document_id": document_id,
                "document_name": document_name,
                "page": page_number,
                "chunk_number": chunk_number,
                "department": department,
                "version": version
            }
        }

        documents.append(document)


# -----------------------------------
# 5. Display results
# -----------------------------------

print("Total chunks:", len(documents))

for i, document in enumerate(documents[:5], start=1):

    print(f"\n--- Chunk {i} ---")

    print("Text:")
    print(document["text"])

    print("\nMetadata:")
    print(document["metadata"])

