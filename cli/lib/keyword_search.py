import string
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
)

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []

    # Tokenize and normalize the query
    query_tokens = tokenize(remove_punctuation(query.lower()))

    for movie in movies:
        # Tokenize and normalize the movie title
        title_tokens = tokenize(remove_punctuation(movie["title"].lower()))

        # Check if at least one query token matches any title token
        if any(query_token in title_tokens for query_token in query_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    return results

def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation from a string using translate() and str.maketrans().

    Args:
        text: The input string to clean

    Returns:
        The string with all punctuation removed

    Example:
        >>> remove_punctuation("Hello, World!")
        "Hello World"
    """
    # Create a translation table that maps each punctuation character to None
    translator = str.maketrans('', '', string.punctuation)

    # Apply the translation to remove punctuation
    return text.translate(translator)

def tokenize(text: str) -> list[str]:
    """
    Tokenize a string into words by splitting on whitespace and removing any empty tokens.

    Args:
        text: The input string to tokenize

    Returns:
        A list of words in the input string

    Example:
        >>> tokenize("Hello World")
        ["Hello", "World"]
    """
    return [token for token in text.split() if token]
