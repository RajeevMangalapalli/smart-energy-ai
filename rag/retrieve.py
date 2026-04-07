# src/rag/retriever.py
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


VECTOR_DB   = Path("smart-energy-ai/data/processed/vectorstore")
EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def retrieve(query: str, top_k: int = 3) -> list[dict]:

    client     = chromadb.PersistentClient(path=str(VECTOR_DB))
    collection = client.get_collection("energy_knowledge")

    # Embed the query with the same model used during ingestion
    query_embedding = EMBED_MODEL.encode(query).tolist()

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = top_k
    )

    # Package results into a clean list of dicts
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text"     : results["documents"][0][i],
            "source"   : results["metadatas"][0][i]["source"],
            "distance" : results["distances"][0][i]   # lower = more similar
        })

    return chunks


def is_relevant(chunks: list[dict], threshold: float = 0.7) -> bool:
    """
    Cosine distance in ChromaDB ranges 0 (identical) to 2 (opposite).
    If the best match is still far away, the query is out of scope.
    """
    if not chunks:
        return False
    return chunks[0]["distance"] < threshold



def generate_answer(query: str, chunks: list[dict], context: str = "") -> dict:
    context_block = ""
    for i, chunk in enumerate(chunks):
        context_block += f"[{i+1}] (Source: {chunk['source']})\n{chunk['text']}\n\n"

    forecast_section = ""
    if context:
        forecast_section = f"Forecast Data:\n{context}\n\n"

    prompt = f"""You are an expert assistant for an energy forecasting system.
Answer the user's question using the context and forecast data provided below.
If the answer cannot be found in the context, say: "I don't have relevant information to answer this question."
Always end your answer with a 'Sources:' line listing which context numbers you used.

{forecast_section}Knowledge Base Context:
{context_block}
Question: {query}

Answer:"""

    response = ollama.chat(
        model    = "mistral",
        messages = [{"role": "user", "content": prompt}]
    )

    answer_text   = response["message"]["content"]
    cited_sources = []
    for i, chunk in enumerate(chunks):
        if f"[{i+1}]" in answer_text:
            cited_sources.append(chunk["source"])
    
    cited_sources = list(set(cited_sources)) #Prevent the duplicate source in the citations
    return {
        "answer"  : answer_text,
        "sources" : cited_sources,
        "chunks"  : chunks
    }



def ask(query: str, context: str = "") -> dict:
    print(f"\nQuery: {query}")
    print("─" * 50)

    chunks = retrieve(query, top_k=3)

    if not is_relevant(chunks):
        return {
            "answer"  : "I don't have relevant information to answer this question.",
            "sources" : [],
            "chunks"  : []
        }

    result = generate_answer(query, chunks, context=context)
    print(f"Answer:\n{result['answer']}")
    print(f"\nCited sources: {result['sources']}")
    return result

if __name__ == "__main__":
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue
        ask(query)