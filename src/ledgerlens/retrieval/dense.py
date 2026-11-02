"""Dense retrieval in Qdrant. Only ids and filterable metadata live in Qdrant;
chunk text stays in the SQLite store."""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from ledgerlens.retrieval.embed import EMBED_DIM, embed_documents, embed_query
from ledgerlens.retrieval.filters import SearchFilters
from ledgerlens.schemas import Chunk, ChunkStrategy

_INDEXED = {
    "ticker": models.PayloadSchemaType.KEYWORD,
    "form": models.PayloadSchemaType.KEYWORD,
    "filing_id": models.PayloadSchemaType.KEYWORD,
    "doc_name": models.PayloadSchemaType.KEYWORD,
    "fiscal_year": models.PayloadSchemaType.INTEGER,
}


def collection_name(strategy: ChunkStrategy) -> str:
    return f"ledgerlens_{strategy}"


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ledgerlens:{chunk_id}"))


def to_qdrant_filter(filters: SearchFilters) -> models.Filter | None:
    conditions: list[models.Condition] = []
    for key, value in filters.model_dump().items():
        if value is None:
            continue
        expected = value.upper() if key == "ticker" else value
        conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=expected)))
    return models.Filter(must=conditions) if conditions else None


class DenseIndex:
    def __init__(self, client: QdrantClient, strategy: ChunkStrategy) -> None:
        self.client = client
        self.collection = collection_name(strategy)

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
            # Keep vectors on disk-backed memmap so the container stays small.
            on_disk_payload=True,
        )
        for field, schema in _INDEXED.items():
            self.client.create_payload_index(self.collection, field, field_schema=schema)

    def add(self, chunks: list[Chunk]) -> None:
        vectors = embed_documents([c.text for c in chunks])
        points = [
            models.PointStruct(
                id=point_id(c.chunk_id),
                vector=v,
                payload={
                    "chunk_id": c.chunk_id,
                    "filing_id": c.filing_id,
                    "ticker": c.ticker,
                    "form": c.form,
                    "fiscal_year": c.fiscal_year,
                    "doc_name": c.doc_name,
                },
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points=points, wait=True)

    def search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[tuple[str, float]]:
        result = self.client.query_points(
            self.collection,
            query=embed_query(query),
            query_filter=to_qdrant_filter(filters or SearchFilters()),
            limit=k,
            with_payload=["chunk_id"],
        )
        return [(str(p.payload["chunk_id"]), float(p.score)) for p in result.points if p.payload]
