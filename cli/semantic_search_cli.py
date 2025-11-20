#!/usr/bin/env python3

import argparse
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    embed_query_text,
    semantic_search,
    chunk_text,
    semantic_chunk_text,
    embed_chunks,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("verify", help="Verify the model is loaded correctly")

    embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text string"
    )
    embed_parser.add_argument("text", type=str, help="Text to embed")

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate an embedding for a single query string"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to embed")
    subparsers.add_parser(
        "verify_embeddings", help="Verify the embeddings are loaded correctly"
    )

    search_parser = subparsers.add_parser(
        "search", help="Search for documents similar to a query"
    )
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument(
        "--limit", type=int, help="Number of results to return", default=5
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunk a text string into smaller segments"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, help="Size of each chunk", default=200
    )
    chunk_parser.add_argument(
        "--overlap", type=int, help="Number of words to overlap between chunks", default=40
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk a text string into smaller segments using semantic search"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, help="Size of each chunk", default=4
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, help="Number of words to overlap between chunks", default=0
    )

    subparsers.add_parser(
        "embed_chunks", help="Generate embeddings for a list of documents"
    )

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
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
