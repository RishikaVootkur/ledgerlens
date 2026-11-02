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

    def ensure_collection(self) -> bool:
        """Create the collection if needed; return True when it was created (empty)."""
        created = False
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(
                    size=EMBED_DIM, distance=models.Distance.COSINE, on_disk=True
                ),
                # Vectors and payload both memory-mapped from disk: the container stays small.
                on_disk_payload=True,
            )
            created = True
        # Idempotent, so a run that died right after creating the collection still
        # ends up with every filter field indexed.
        for field, schema in _INDEXED.items():
            self.client.create_payload_index(self.collection, field, field_schema=schema)
        return created

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.client.delete(
                self.collection,
                points_selector=models.PointIdsList(points=[point_id(c) for c in chunk_ids]),
            )

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
