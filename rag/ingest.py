#Goal of this file is to Load all documents from /rag/documents
#Split them into chunks (300-500 chunks)

"""
GENERAL STEPS IN THIS FILE

1) LOADING THE FILES
2) CHUNKING
3) EMBEDDING
4) STORING IT IN THE VECTOR STORE

"""



import os
from langchain_text_splitters import RecursiveCharacterTextSplitter as RCS
from sentence_transformers import SentenceTransformer
from pathlib import Path
import chromadb
import json



DATASET_DESCRIPTION_PATH = Path("smart-energy-ai/rag/documents/dataset_description.txt")
ENERGY_FORECASTING_CONCEPTS = Path("smart-energy-ai/rag/documents/energy_forecasting_concepts.txt")
MODEL_ASSUMPTIONS = Path("smart-energy-ai/rag/documents/model_assumptions_and_features.txt")

VECTOR_DB = Path("smart-energy-ai/data/processed/vectorstore")
VECTOR_DB.mkdir(parents=True, exist_ok = True)
MAPPING_PATH = Path("smart-energy-ai/data/processed/chunk_mapping.json")

DOC_PATHS = [DATASET_DESCRIPTION_PATH,ENERGY_FORECASTING_CONCEPTS,MODEL_ASSUMPTIONS]

# Loading the text documents

def load_docs(DOC_PATHS : list[Path]) -> list[dict]:
    documents = []
    for file in DOC_PATHS:
        text = file.read_text(encoding="utf-8")

        documents.append(
            {
                "source" : file.name,
                "text" : text
            }
        )
        print(f"Loaded: {file.name} ({len(text)} chars)")
    
    return documents


#Chunking

def chunk_docs(docs : list[dict]) -> list[dict]:

    text_splitter = RCS(
        chunk_size=300,
        chunk_overlap=50,
        length_function = len,
        separators=["\n\n", "\n", "."," "]
    )

    chunks =[]
    
    for doc in docs:
        splits = text_splitter.split_text(doc["text"])

        for i,split in enumerate(splits):
            chunks.append({
                "chunk_id": f"{doc["source"]}_chunk{i}",
                "source" : doc["source"],
                "chunk" : split,
            })
    
    print(f"Total Chunks created: {len(chunks)}")
    return chunks


# Embedding the chunks


def embedding_chunks(chunks : list[dict]) -> list[list]:

    chunks_pre_embedding = [c["chunk"] for c in chunks]

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(chunks_pre_embedding, show_progress_bar=True)

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()

    return chunks


# store in the vector database

def store_in_vecdb(chunks: list[dict]) -> chromadb.Collection:
    client = chromadb.PersistentClient(path = str(VECTOR_DB))

    try:
        client.delete_collection("energy_knowledge")
    
    except:
        pass
    

    collection = client.create_collection(
        name = "energy_knowledge",
        metadata={"hnsw:space":"cosine"} #cosine similarity for text

    )

    collection.add(
        ids = [c["chunk_id"] for c in chunks],
        documents = [c["chunk"] for c in chunks],
        embeddings = [c["embedding"] for c in chunks],
        metadatas = [{"source": c["source"]} for c in chunks]
    )
    print(f"\nStored {len(chunks)} chunks in the ChromaDB at {VECTOR_DB}")
    return collection

# Save chunk mapping
def save_chunk_mapping(chunks: list[dict]):
    mapping ={
        c["chunk_id"]:{
            "source": c["source"],
            "text" : c["chunk"]
        }
        for c in chunks
    }
    with open(MAPPING_PATH, "w", encoding = "utf-8") as f:
        json.dump(mapping, f, indent =2)
    print(f"Chunk mapping saved")



if __name__ == "__main__":

    documents = load_docs(DOC_PATHS)
    chunks = chunk_docs(documents)
    chunks = embedding_chunks(chunks)
    store_in_vecdb(chunks)
    save_chunk_mapping(chunks)

    print("Sucessful")
