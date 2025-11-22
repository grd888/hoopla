from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re
import json
from .search_utils import (load_movies, SCORE_PRECISION)


class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
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
        self.documents = documents
        self.document_map = {}

        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        scores = []
        for index, embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(embedding, query_embedding)
            scores.append((similarity, self.documents[index]))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {"score": score, "title": doc["title"], "description": doc["description"]}
            for score, doc in scores[:limit]
        ]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []
        for doc_idx, doc in enumerate(documents):
            if doc["description"] is None:
                continue
            chunks = semantic_chunk_text(doc["description"], max_chunk_size=4, overlap=1)
            all_chunks.extend(chunks)
            for chunk_idx, chunk in enumerate(chunks):
                chunk_metadata.append({
                    "movie_idx": doc_idx, 
                    "chunk_idx": chunk_idx, 
                    "total_chunks": len(chunks)
                })
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings
        
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            with open("cache/chunk_metadata.json", "r") as f:
                metadata_json = json.load(f)
                self.chunk_metadata = metadata_json["chunks"]
            if len(self.chunk_embeddings) == metadata_json["total_chunks"]:
                return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query_embedding = self.generate_embedding(query)
        scores: list[dict] = []
        for index, embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(embedding, query_embedding)
            scores.append({
                "score": similarity,
                "movie_idx": self.chunk_metadata[index]["movie_idx"],
                "chunk_idx": self.chunk_metadata[index]["chunk_idx"],
            })
        movie_to_score_dict: dict[int, dict] = {}
        for score in scores:
            if score["movie_idx"] not in movie_to_score_dict or movie_to_score_dict[score["movie_idx"]]["score"] < score["score"]:
                movie_to_score_dict[score["movie_idx"]] = score
        scores_sorted = sorted(movie_to_score_dict.values(), key=lambda x: x["score"], reverse=True)
        scores_filtered = scores_sorted[:limit]
        return [
            {
                "id": self.documents[score["movie_idx"]]["id"],
                "title": self.documents[score["movie_idx"]]["title"],
                "document": self.documents[score["movie_idx"]]["description"][:100],
                "score": round(score["score"], SCORE_PRECISION),
                "metadata": self.documents[score["movie_idx"]].get("metadata", {}),
            }
            for score in scores_filtered
        ]


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
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_search(query, limit):
    semantic_search = SemanticSearch()
    documents = load_movies()
    semantic_search.load_or_create_embeddings(documents)
    results = semantic_search.search(query, limit)
    for index, result in enumerate(results):
        print(
            f"{index + 1}. {result['title']} ({result['score']:.4f})\n{result['description']}\n"
        )


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    words = text.split(" ")
    chunks = []

    # Ensure a valid step size when using overlap
    if overlap > 0 and overlap < chunk_size:
        step = chunk_size - overlap
    else:
        step = chunk_size

    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i : i + chunk_size]))

    print(f"Chunking {len(text)} characters")
    for index, chunk in enumerate(chunks):
        print(f"{index + 1}. {chunk}")

    return chunks

def semantic_chunk_text(text: str, max_chunk_size: int = 200, overlap: int = 40) -> list[str]:
    text = text.strip()
    if not text:
        return []

    pattern = r"(?<=[.!?])\s+"
    sentences = [s for s in re.split(pattern, text) if s.strip()]
    
    if len(sentences) == 1 and not re.search(r"[.!?]$", sentences[0]):
        sentences = [text]

    sentences = [s.strip() for s in sentences if s.strip()]

    # Each chunk should contain up to max_chunk_size sentences.
    # Support overlap by number of sentences.
    chunks = []
    
    # Ensure a valid step size when using overlap
    if overlap > 0 and overlap < max_chunk_size:
        step = max_chunk_size - overlap
    else:
        step = max_chunk_size

    for i in range(0, len(sentences), step):
        if i > 0 and i + overlap >= len(sentences):
            continue
        chunks.append(" ".join(sentences[i : i + max_chunk_size]))
    # Return a list of chunk strings.
    print(f"Semantically chunking {len(text)} characters")
    for index, chunk in enumerate(chunks):
        print(f"{index + 1}. {chunk}")

    return chunks

def embed_chunks():
    semantic_search = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked(query, limit):
    semantic_search = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_chunk_embeddings(documents)
    results = semantic_search.search_chunks(query, limit)
    for index, result in enumerate(results):
        print(
            f"\n{index + 1}. {result['title']} (score: {result['score']:.4f})"
        )
        print(f"   {result['document']}...")