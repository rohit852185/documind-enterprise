"""
Week 1 – Ingestion Pipeline
PDF → Chunk → Embedding → Pinecone
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

PDF_DIR = "data/pdfs"
INDEX_NAME = "documind-week1"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_ENV")  # example: us-east-1

# Safety checks
if not OPENAI_API_KEY:
    raise ValueError(" OPENAI_API_KEY not found in .env")

if not PINECONE_API_KEY:
    raise ValueError(" PINECONE_API_KEY not found in .env")

if not PINECONE_REGION:
    raise ValueError(" PINECONE_ENV (region) not found in .env")


# Step 1: Load PDFs
def load_pdfs():
    documents = []

    if not os.path.exists(PDF_DIR):
        raise FileNotFoundError(f" Folder not found: {PDF_DIR}")

    for file in os.listdir(PDF_DIR):
        if file.lower().endswith(".pdf"):
            path = os.path.join(PDF_DIR, file)
            print(f" Loading PDF: {file}")

            loader = UnstructuredPDFLoader(path, mode="elements")
            docs = loader.load()

            # add filename to metadata
            for d in docs:
                d.metadata["source"] = file

            documents.extend(docs)

    if not documents:
        raise ValueError("No PDFs found in data/pdfs")

    return documents


# Step 2: Chunk documents
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

# Step 3: Initialize Pinecone
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = pc.list_indexes().names()

    if INDEX_NAME not in existing_indexes:
        print("🧠 Creating Pinecone index...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_REGION
            )
        )

    return pc.Index(INDEX_NAME)

# Step 4: Run ingestion
def run_ingestion():
    print("\n Starting ingestion pipeline...\n")

    # Load PDFs
    docs = load_pdfs()
    print(f"Total pages loaded: {len(docs)}")

    # Chunking
    chunks = chunk_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    # Embeddings
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Pinecone
    index = init_pinecone()

    # Batch upsert (VERY IMPORTANT for large PDFs)
    batch_size = 100
    vectors = []

    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_query(chunk.page_content)

        metadata = {
            "source": chunk.metadata.get("source"),
            "page": chunk.metadata.get("page"),
            "text": chunk.page_content[:500]  # small preview
        }

        vectors.append((str(i), vector, metadata))

        # upsert in batches
        if len(vectors) >= batch_size:
            index.upsert(vectors)
            vectors = []

    # upsert remaining
    if vectors:
        index.upsert(vectors)

    print("Ingestion complete. Data stored in Pinecone.\n")

# --------------------------------------------------
if __name__ == "__main__":
    run_ingestion()