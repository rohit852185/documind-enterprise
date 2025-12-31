"""
Week 1 – Retrieval Test
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

load_dotenv()

INDEX_NAME = "documind-week1"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY")
)

query = "How do I get my money back?"

query_vector = embeddings.embed_query(query)

result = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True
)

print("\nTop matching chunks:\n")
for match in result["matches"]:
    print("Score:", match["score"])
    print("Metadata:", match["metadata"])
    print("-" * 40)