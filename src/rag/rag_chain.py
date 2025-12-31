import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone

from src.utils.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_ENV")

INDEX_NAME = "documind-week1"  # SAME AS INGEST.PY

# ---- Initialize Pinecone ----
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ---- Embeddings ----
embeddings = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

# ---- LLM ----
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)

# ---- MAIN RAG FUNCTION ----
def run_rag(query):
    """Retrieve context from Pinecone → Send to LLM → Get final answer"""

    # 1️⃣ Embed the question
    query_vector = embeddings.embed_query(query)

    # 2️⃣ Perform similarity search
    results = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )

    # 3️⃣ Build context string
    context_texts = []
    for match in results["matches"]:
        text = match["metadata"].get("text", "")
        source = match["metadata"].get("source", "Unknown")
        page = match["metadata"].get("page", "N/A")

        formatted = f"{text}\n(Source: {source}, Page: {page})"
        context_texts.append(formatted)

    context = "\n\n".join(context_texts)

    # 4️⃣ Prepare final prompt for the LLM
    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=query,
        context=context
    )

    # 5️⃣ Generate final answer
    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ])

    return response.content
