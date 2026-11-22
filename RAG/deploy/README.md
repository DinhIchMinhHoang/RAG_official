# RAG storage services (local dev)

Postgres (metadata, chunks, ingest status, API usage), MinIO (raw files, page images, parsed blocks),
Qdrant (vectors; client code comes later).

## Start / stop
    cd RAG/deploy
    cp .env.example .env        # dev defaults: change passwords/keys for anything shared
    docker compose up -d
    docker compose down         # stop (data kept)

## Check / init (from RAG/)
    python -m store.doctor          # one OK/FAIL line per service, exit 0 only if all OK
    python -m store.doctor --init   # also create schema + bucket

Python reads RAG_PG_*, RAG_MINIO_*, RAG_QDRANT_* env vars (see store/settings.py); defaults match .env.example.

## Ports (127.0.0.1 only)
Postgres 5432, MinIO API 9000 and console 9001, Qdrant 6333 (HTTP) and 6334 (gRPC).

## Data
`RAG/.rag_data/{postgres,minio,qdrant}` (git-ignored). Delete it to reset everything.
MinIO image is `latest`: pin a RELEASE tag before production.
