import string
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords
)


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
    return filtered_words


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
    movies = load_movies()
    results = []

    for movie in movies:
        query_tokens = tokenize_text(query)
        # Tokenize and normalize the movie title
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
              
    return results
