"""
#retireve.py
User query -> Chunk(?) -> generate embeddings -> Do a comparision with the vector-database -> Return the top 3 query answers

"""


from langchain_text_splitters import RecursiveCharacterTextSplitter as RCS
from pathlib import Path
import chromadb


VECTOR_DB = Path("smart-energy-ai/data/processed/vectorstore")

#User query
user_query1 = str(input("Enter your query:"))
user_query2 = str(input("Enter your query:"))
user_query3 = str(input("Enter your query:"))

"""

#Embedding
def embedding_query(user_query : str) -> list:
    
    user_query = str(input("Enter a query: "))
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(user_query, show_progress_bar=True)

    return embeddings
"""

# Similarity Search
def similarity_search():
    client = chromadb.PersistentClient(path = str(VECTOR_DB))

    collection = client.get_or_create_collection(name ="energy_knowledge")

    results = collection.query(
        query_texts =[user_query1, user_query2, user_query3],
        n_results = 2
    )
    return results

