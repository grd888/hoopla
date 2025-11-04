import math
import os
import pickle
import string
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    BM25_K1,
    BM25_B,
    load_movies,
    load_stopwords
)

class InvertedIndex:
    def __init__(self):
        # dictionary mapping tokens(strings) to sets of document IDs(integers)
        self.index = defaultdict(set)
        # dictionary mapping document IDs to their corresponding document objects
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: defaultdict(Counter) = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        
    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)
    # get the set of documents for a given token
    # and return them as a list, sorted in ascending order
    # by document ID.
    def get_documents(self, term: str) -> list[int]:
        doc_ids = list(self.index.get(term, set()))
        return sorted(doc_ids)

    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError(f"Term must be a single token, got {len(tokens)} tokens")
        
        token = tokens[0]
        return self.term_frequencies[doc_id][token]
    
    def get_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError(f"Term must be a single token, got {len(tokens)} tokens")
        
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))
    
    def get_bm25_idf(self, term: str) -> float:
        """Calculate BM25 IDF score for a term.
        
        Args:
            term: The term to calculate BM25 IDF for.
            
        Returns:
            The BM25 IDF score using the formula: log((N - df + 0.5) / (df + 0.5) + 1)
            
        Raises:
            ValueError: If the term tokenizes to more than one token.
        """
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError(f"Term must be a single token, got {len(tokens)} tokens")
        
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * doc_length / avg_doc_length
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def bm25(self, doc_id, term):
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term)
        return idf * tf
    
    def bm25_search(self, query, limit):
        query_tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            score = 0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)   
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)
            
    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)

def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    

def preprocess_text(text: str) -> str:
    """Preprocess text by converting to lowercase and removing punctuation.
    
    Args:
        text: The input text string to preprocess.
        
    Returns:
        The preprocessed text with lowercase characters and no punctuation.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def tokenize_text(text: str) -> list[str]:
    """Tokenize text into words, removing stopwords and empty tokens.
    
    Preprocesses the text, splits it into tokens, filters out empty strings,
    and removes common stopwords.
    
    Args:
        text: The input text string to tokenize.
        
    Returns:
        A list of filtered tokens with stopwords removed.
    """
    stemmer = PorterStemmer()
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return stemmed_words


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    """Check if any query token is a substring of any title token.
    
    Args:
        query_tokens: List of tokens from the search query.
        title_tokens: List of tokens from a movie title.
        
    Returns:
        True if any query token is found within any title token, False otherwise.
    """
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    """Search for movies matching the query tokens.
    
    Searches through the movie database for titles containing any of the query tokens.
    Returns up to the specified limit of matching movies.
    
    Args:
        query: The search query string.
        limit: Maximum number of results to return. Defaults to DEFAULT_SEARCH_LIMIT.
        
    Returns:
        A list of movie dictionaries that match the search query.
    """
    results = []
    idx = InvertedIndex()
    idx.load()
    query_tokens = tokenize_text(query)
    for token in query_tokens:
        doc_ids = idx.get_documents(token)
        for doc_id in doc_ids:
            results.append(idx.docmap[doc_id])
            if len(results) >= limit:
                return results
    return results

def tf_command(doc_id: int, term: str) -> int:
    """Get the term frequency for a term in a specific document.
    
    Args:
        doc_id: The document ID to check.
        term: The term to look up.
        
    Returns:
        The term frequency count, or 0 if the term doesn't exist in the document.
    """
    idx = InvertedIndex()
    idx.load()
    # Tokenize the term to match how it's stored in the index
    tokens = tokenize_text(term)
    if not tokens:
        return 0
    # Use the first token (terms should be single words)
    return idx.get_tf(doc_id, tokens[0])
   
def idf_command(term: str) -> float:
    """Get the inverse document frequency for a term."""
    idx = InvertedIndex()
    idx.load()
    tokens = tokenize_text(term)
    if not tokens:
        return 0
    return idx.get_idf(tokens[0])

def tfidf_command(doc_id: int, term: str) -> float:
    """Calculate the TF-IDF score for a term in a specific document.
    
    Args:
        doc_id: The document ID to check.
        term: The term to calculate TF-IDF for.
        
    Returns:
        The TF-IDF score (TF * IDF).
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf_idf(doc_id, term)

def bm25_idf_command(term: str) -> float:
    """Get the BM25 IDF score for a term.
    
    Args:
        term: The term to calculate BM25 IDF for.
        
    Returns:
        The BM25 IDF score.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, term, k1, b)

def bm25_search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[tuple[int, float]]:
    """Search for movies using BM25 scoring.
    
    Args:
        query: The search query string.
        limit: Maximum number of results to return. Defaults to DEFAULT_SEARCH_LIMIT.
        
    Returns:
        A list of tuples containing (doc_id, score) sorted by score in descending order.
    """
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)
    