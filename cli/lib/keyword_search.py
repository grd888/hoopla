import os
import pickle
import string
from collections import defaultdict
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
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str) -> list[int]:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
    # get the set of documents for a given token
    # and return them as a list, sorted in ascending order
    # by document ID.
    def get_documents(self, term: str) -> list[int]:
        doc_ids = list(self.index.get(term, set()))
        return sorted(doc_ids)

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

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
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
   
