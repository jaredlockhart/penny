"""One-off script to recompute all entity and fact embeddings.

Run inside the penny Docker container:
    docker compose run --rm penny python scripts/recompute_embeddings.py

Uses OLLAMA_EMBEDDING_MODEL (or OLLAMA_BACKGROUND_MODEL as fallback)
to regenerate embeddings for every entity and fact in the database.
"""

from __future__ import annotations

import asyncio
import os
import sys

from penny.config import Config
from penny.database import Database
from penny.ollama.client import OllamaClient
from penny.ollama.embeddings import build_entity_embed_text, serialize_embedding

BATCH_SIZE = 50


async def recompute_all() -> None:
    config = Config.load()
    db = Database(config.db_path)

    model = config.ollama_embedding_model or config.ollama_background_model
    if not model:
        print("No embedding model configured (OLLAMA_EMBEDDING_MODEL or OLLAMA_BACKGROUND_MODEL)")
        sys.exit(1)

    client = OllamaClient(
        api_url=config.ollama_api_url,
        model=model,
        db=db,
    )

    try:
        # --- Facts ---
        all_facts = db.get_facts_without_embeddings(limit=999999)
        total_facts = len(all_facts)
        print(f"Facts to embed: {total_facts}")

        for i in range(0, total_facts, BATCH_SIZE):
            batch = all_facts[i : i + BATCH_SIZE]
            texts = [f.content for f in batch]
            vecs = await client.embed(texts)
            for fact, vec in zip(batch, vecs, strict=True):
                assert fact.id is not None
                db.update_fact_embedding(fact.id, serialize_embedding(vec))
            print(f"  Facts: {min(i + BATCH_SIZE, total_facts)}/{total_facts}")

        # --- Entities ---
        all_entities = db.get_entities_without_embeddings(limit=999999)
        total_entities = len(all_entities)
        print(f"Entities to embed: {total_entities}")

        for i in range(0, total_entities, BATCH_SIZE):
            batch = all_entities[i : i + BATCH_SIZE]
            texts = []
            for entity in batch:
                assert entity.id is not None
                facts = db.get_entity_facts(entity.id)
                texts.append(
                    build_entity_embed_text(
                        entity.name, [f.content for f in facts], entity.tagline
                    )
                )
            vecs = await client.embed(texts)
            for entity, vec in zip(batch, vecs, strict=True):
                assert entity.id is not None
                db.update_entity_embedding(entity.id, serialize_embedding(vec))
            print(f"  Entities: {min(i + BATCH_SIZE, total_entities)}/{total_entities}")

        print("Done.")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(recompute_all())
