#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, semantic_search


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("verify", help="Verify the model is loaded correctly")
    
    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a single text string")
    embed_parser.add_argument("text", type=str, help="Text to embed")
    
    embed_query_parser = subparsers.add_parser("embedquery", help="Generate an embedding for a single query string")
    embed_query_parser.add_argument("query", type=str, help="Query to embed")
    subparsers.add_parser("verify_embeddings", help="Verify the embeddings are loaded correctly")
    
    search_parser = subparsers.add_parser("search", help="Search for documents similar to a query")
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument("--limit", type=int, help="Number of results to return", default=5)
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "search":
            semantic_search(args.query, args.limit)    
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
