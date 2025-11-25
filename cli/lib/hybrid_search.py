import os 

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

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
    raise NotImplementedError("Weighted hybrid search is not implemented yet")

  def rrf_search(self, query, limit=10):
    raise NotImplementedError("RRF hybrid search is not implemented yet")

def normalize(scores):
  if len(scores) == 0:
    return
  if min(scores) == max(scores):
    # print a list of 1.0 values with the same length as scores
    print([1.0] * len(scores))
  else:
    min_score = min(scores)
    max_score = max(scores)
    range_score = max_score - min_score
    normalized_scores = list(map(lambda x: (x - min_score) / range_score, scores)) 
    for score in normalized_scores:
      print(f"* {score:.4f}")
