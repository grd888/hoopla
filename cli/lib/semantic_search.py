from sentence_transformers import SentenceTransformer
import numpy as np
import os
from .search_utils import load_movies
class SemanticSearch:
  def __init__(self):
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    self.embeddings = None
    self.documents = None
    self.document_map = {}
    
  def generate_embedding(self, text: str) -> list[float]:
    embeddings = self.model.encode([text])
    return embeddings[0]
  
  def build_embeddings(self, documents):
    self.documents = documents
    for doc in documents:
      self.document_map[doc["id"]] = doc
      
    doc_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
    self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
    np.save("cache/movie_embeddings.npy", self.embeddings)
    return self.embeddings
  
  def load_or_create_embeddings(self, documents):
    if os.path.exists("cache/movie_embeddings.npy"):
      self.embeddings = np.load("cache/movie_embeddings.npy")
      if len(self.embeddings) == len(documents):
        return self.embeddings
      
    return self.build_embeddings(documents)
    
def verify_model():
  semantic_search = SemanticSearch()
  print(f"Model loaded: {semantic_search.model}")
  print(f"Max sequence length: {semantic_search.model.max_seq_length}")
  
def embed_text(text: str):
  semantic_search = SemanticSearch()
  embedding = semantic_search.generate_embedding(text)
  print(f"Text: {text}")
  print(f"First 3 dimensions: {embedding[:3]}")
  print(f"Dimensions: {embedding.shape[0]}")
  
def embed_query_text(query: str) -> None:
  semantic_search = SemanticSearch()
  embedding = semantic_search.generate_embedding(query)
  print(f"Query: {query}")
  print(f"First 5 dimensions: {embedding[:5]}")
  print(f"Dimensions: {embedding.shape}")
  
def verify_embeddings():
  semantic_search = SemanticSearch()
  documents = load_movies()
  embeddings = semantic_search.load_or_create_embeddings(documents)
  print(f"Number of docs:   {len(documents)}")
  print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
  