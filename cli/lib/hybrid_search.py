import os
import time

from dotenv import load_dotenv
from google import genai

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .query_enhancement import enhance_query

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash"


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.idx = InvertedIndex()

        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        # Get 500x the limit to ensure enough results to work with
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        # Extract scores for normalization
        bm25_scores = [score for _, score in bm25_results]
        semantic_scores = [result["score"] for result in semantic_results]

        # Normalize scores
        bm25_normalized = normalize(bm25_scores)
        semantic_normalized = normalize(semantic_scores)

        # Create dictionary mapping doc IDs to documents and scores
        doc_scores = {}

        # Add BM25 results
        for i, (doc_id, _) in enumerate(bm25_results):
            doc_scores[doc_id] = {
                "document": self.idx.docmap[doc_id],
                "keyword_score": bm25_normalized[i] if bm25_normalized else 0.0,
                "semantic_score": 0.0,
            }

        # Add semantic results
        for i, result in enumerate(semantic_results):
            doc_id = result["id"]
            if doc_id in doc_scores:
                doc_scores[doc_id]["semantic_score"] = (
                    semantic_normalized[i] if semantic_normalized else 0.0
                )
            else:
                doc_scores[doc_id] = {
                    "document": self.documents[
                        next(
                            j for j, d in enumerate(self.documents) if d["id"] == doc_id
                        )
                    ],
                    "keyword_score": 0.0,
                    "semantic_score": semantic_normalized[i]
                    if semantic_normalized
                    else 0.0,
                }

        # Calculate hybrid score for each document
        for doc_id in doc_scores:
            doc_scores[doc_id]["hybrid_score"] = hybrid_score(
                doc_scores[doc_id]["keyword_score"],
                doc_scores[doc_id]["semantic_score"],
                alpha,
            )

        # Sort by hybrid score descending and return top results
        sorted_results = sorted(
            doc_scores.items(), key=lambda x: x[1]["hybrid_score"], reverse=True
        )[:limit]

        return [
            {
                "id": doc_id,
                "document": data["document"],
                "keyword_score": data["keyword_score"],
                "semantic_score": data["semantic_score"],
                "hybrid_score": data["hybrid_score"],
            }
            for doc_id, data in sorted_results
        ]

    def rrf_search(self, query, k=60, limit=10, enhance=None, rerank_method=None):
        if enhance in ["spell", "rewrite", "expand"]:
            enhanced_query = enhance_query(query, enhance)
            print(f"Enhanced query ({enhance}): '{query}' -> '{enhanced_query}'\n")
            query = enhanced_query

        # Gather 5x results if reranking
        search_limit = limit * 5 if rerank_method == "individual" else limit

        bm25_results = self._bm25_search(query, search_limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, search_limit * 500)

        sorted_bm25_results = sorted(bm25_results, key=lambda x: x[1], reverse=True)
        sorted_semantic_results = sorted(
            semantic_results, key=lambda x: x["score"], reverse=True
        )

        # Create a dictionary mapping document IDs to the documents themselves and their BM25 and semantic ranks (not scores)
        doc_ranks = {}
        for i, (doc_id, _) in enumerate(sorted_bm25_results):
            doc_ranks[doc_id] = {
                "document": self.idx.docmap[doc_id],
                "bm25_rank": i + 1,
                "semantic_rank": 0,
            }
        for i, result in enumerate(sorted_semantic_results):
            doc_id = result["id"]
            if doc_id in doc_ranks:
                doc_ranks[doc_id]["semantic_rank"] = i + 1
            else:
                doc_ranks[doc_id] = {
                    "document": self.documents[
                        next(
                            j for j, d in enumerate(self.documents) if d["id"] == doc_id
                        )
                    ],
                    "bm25_rank": 0,
                    "semantic_rank": i + 1,
                }

        # Calculate RRF score for each document
        for doc_id in doc_ranks:
            doc_ranks[doc_id]["rrf_score"] = rrf_score(
                (doc_ranks[doc_id]["bm25_rank"] + doc_ranks[doc_id]["semantic_rank"]), k
            )

        # Sort by RRF score descending and return top results
        sorted_results = sorted(
            doc_ranks.items(), key=lambda x: x[1]["rrf_score"], reverse=True
        )[:search_limit]

        results = [
            {
                "id": doc_id,
                "document": data["document"],
                "bm25_rank": data["bm25_rank"],
                "semantic_rank": data["semantic_rank"],
                "rrf_score": data["rrf_score"],
            }
            for doc_id, data in sorted_results
        ]

        # Apply reranking if method is specified
        if rerank_method == "individual":
            print(f"Reranking top {limit} results using individual method...")
            results = self._rerank_individual(query, results, limit)

        return results[:limit]

    def _rerank_individual(self, query, results, limit):
        """Rerank results using individual LLM prompts for each document."""
        for result in results:
            doc = result["document"]
            prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
            response = client.models.generate_content(model=model, contents=prompt)
            try:
                score = float((response.text or "0").strip())
                result["rerank_score"] = min(max(score, 0), 10)  # Clamp to 0-10
            except ValueError:
                result["rerank_score"] = 0.0

            time.sleep(3)  # Avoid rate limiting

        # Sort by rerank score descending
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return results[:limit]


def hybrid_score(keyword_score, semantic_score, alpha):
    return alpha * keyword_score + (1 - alpha) * semantic_score


def normalize(scores):
    if len(scores) == 0:
        return []
    if min(scores) == max(scores):
        return [1.0] * len(scores)
    else:
        min_score = min(scores)
        max_score = max(scores)
        range_score = max_score - min_score
        return list(map(lambda x: (x - min_score) / range_score, scores))


def rrf_score(rank, k=60):
    return 1 / (k + rank)
