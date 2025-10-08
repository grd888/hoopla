import string
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
)


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


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        if remove_punctuation(query.lower()) in remove_punctuation(movie["title"].lower()):
            results.append(movie)
            if len(results) >= limit:
                break
    return results
