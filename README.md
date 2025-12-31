# DocuMind – Enterprise RAG System

DocuMind is an enterprise-style **Retrieval Augmented Generation (RAG)** system designed to process documents, store semantic embeddings, and enable intelligent query-based retrieval.

This repository currently contains **Week 1 – Ingestion Pipeline** implementation.

---

## 🚀 Week 1: Ingestion Pipeline (Completed)

### What is implemented?
The ingestion pipeline takes PDF documents, processes them, and stores vector embeddings in Pinecone for semantic search.

### Pipeline Flow
PDF → Text Extraction → Chunking → Embeddings → Pinecone Vector Store

---

## 🔧 Features Implemented

- PDF loading using **LangChain**
- Recursive text chunking for large documents
- OpenAI embeddings generation
- Pinecone serverless vector database integration
- Metadata enrichment (source file, page number)
- Batch-safe ingestion
- Cost-controlled ingestion (limited chunks for development)

---

## 🛠️ Tech Stack

- **Python 3.11**
- **LangChain**
  - langchain-community
  - langchain-openai
  - langchain-text-splitters
- **OpenAI API** (Embeddings)
- **Pinecone** (Serverless Vector DB)
- **python-dotenv**

---

## 📁 Project Structure
