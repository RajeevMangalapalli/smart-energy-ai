#Goal of this file is to Load all documents from /rag/documents
#Split htem into chunks (300-500 chunks)


import os
from pathlib import Path
import chromadb
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter as RCS

"""
FILE PATHS
"""

DATASET_DESC = Path("smart-energy-ai/rag/documents/dataset_description.txt")
ENERGY_FORECASTING_CONCEPTS = Path("smart-energy-ai/rag/documents/energy_forecasting_concepts.txt")
MODEL_ASSUMPTIONS = Path("smart-energy-ai/rag/documents/model_assumptions_and_features.txt")
DOC_PATHS = [DATASET_DESC,ENERGY_FORECASTING_CONCEPTS,MODEL_ASSUMPTIONS]


#Loading the files
def load_docs(docs_path : list[Path]) -> list[dict]:
    documents =[]

    for file in DOC_PATHS:
        text = file.read_text(encoding='utf-8')
        documents.append({
            "source": file.name,
            "text": text
        })
        print(f"Loaded: {file.name} ({len(text)} chars)")
    
    return documents

# Chunking documents
def chunk_documents(documents: list[dict]) -> list[dict]:
    splitter = RCS(
        chunk_size = 300,
        chunk_overlap = 50,
        separators = ["\n\n", "\n", ".", " "]
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "chunk_id" : f"{doc['source']}__chunk_{i}",
                "source"   : doc["source"],
                "text"     : split
            })
        
    print(f"\n Total chunks created: {len(chunks)}")
    return chunks





