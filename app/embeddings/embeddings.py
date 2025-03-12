from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from db.conection_qdrant import save_embeddings 
import logging

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    return embeddings.flatten().tolist()

def generate_embeddings_from_context_file(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            context = file.read()

        chunks = context.split('\n\n') 

        embeddings = [embed_text(chunk) for chunk in chunks]

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            logging.debug(f"\nChunk [{i}]: {chunk[:100]}...")  
            logging.debug(f"Embedding [{i}]: {embedding[:5]}...")

        # Salvar os embeddings no banco vetorial
        save_embeddings(chunks, embeddings)
        print(f"Embeddings gerados e salvos com sucesso para o arquivo: {file_path}")

        return chunks

    except FileNotFoundError:
        print(f"Arquivo de contexto não encontrado: {file_path}")
        return []
