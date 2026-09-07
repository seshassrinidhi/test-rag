from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Read PDF
pdf_path = "Documents/company_policy.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


# 2. Create recursive text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)


# 3. Split the document
chunks = splitter.split_text(text)


# 4. Display results
print("Number of pages:", len(reader.pages))
print("Total characters:", len(text))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)