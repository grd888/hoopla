import os
import pickle
import string
import math
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer
from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords
)

class InvertedIndex:
    def __init__(self):
        # dictionary mapping tokens(strings) to sets of document IDs(integers)
        self.index = defaultdict(set)
        # dictionary mapping document IDs to their corresponding document objects
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def __add_document(self, doc_id: int, text: str) -> list[int]:
        tokens = tokenize_text(text)
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
    # get the set of documents for a given token
    # and return them as a list, sorted in ascending order
    # by document ID.
    def get_documents(self, term: str) -> list[int]:
        doc_ids = list(self.index.get(term, set()))
        return sorted(doc_ids)

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies.get(doc_id, {}).get(term, 0)
    
    def get_idf(self, term: str) -> float:
        return math.log((len(self.docmap) + 1) / (len(self.get_documents(term)) + 1))
    
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
        N = len(self.docmap)
        df = len(self.get_documents(token))
        
        return math.log((N - df + 0.5) / (df + 0.5) + 1)
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
    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
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
    tokens = tokenize_text(term)
    if not tokens:
        return 0.0
    token = tokens[0]
    tf = idx.get_tf(doc_id, token)
    idf = idx.get_idf(token)
    return tf * idf

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
    