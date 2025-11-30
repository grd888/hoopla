import argparse
from lib.hybrid_search import normalize, HybridSearch
from lib.search_utils import load_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument(
        "scores", type=float, nargs="+", help="List of scores to normalize"
    )

    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Weighted hybrid search"
    )
    weighted_search_parser.add_argument("query", type=str, help="Query to search for")
    weighted_search_parser.add_argument(
        "--alpha", type=float, help="Weight for BM25 score", default=0.5
    )
    weighted_search_parser.add_argument(
        "--limit", type=int, help="Number of results to return", default=5
    )

    rrf_search_parser = subparsers.add_parser("rrf-search", help="RRF hybrid search")
    rrf_search_parser.add_argument("query", type=str, help="Query to search for")
    rrf_search_parser.add_argument(
        "--limit", type=int, help="Number of results to return", default=5
    )
    rrf_search_parser.add_argument(
        "--k", type=int, help="Number of results to return", default=60
    )
    rrf_search_parser.add_argument(
        "--enhance", type=str, choices=["spell", "rewrite"], help="Query enhancement method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            movies = load_movies()
            hybrid_search = HybridSearch(movies)
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            for result in results:
                print(f"{result['document']['title']}")
        case "rrf-search":
            movies = load_movies()
            hybrid_search = HybridSearch(movies)
            results = hybrid_search.rrf_search(args.query, args.k, args.limit, args.enhance)
            for i, result in enumerate(results, 1):
                title = result["document"]["title"]
                rrf_score = result["rrf_score"]
                bm25_rank = result["bm25_rank"]
                semantic_rank = result["semantic_rank"]
                overview = result["document"]["description"][:100] + "..."
                print(f"{i}. {title}")
                print(f"   RRF Score: {rrf_score:.3f}")
                print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"   {overview}")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
