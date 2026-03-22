import json
import logging

import ollama
from pymilvus import MilvusClient

from src.settings.config import settings

logger = logging.getLogger(__name__)


def build_embed_text(item: dict) -> str:
    """Combine content + metadata into a single string for embedding."""
    parts = []
    if item.get("subject"):
        parts.append(f"Subject: {item['subject']}")
    if item.get("chapter"):
        parts.append(f"Chapter: {item['chapter']}")
    if item.get("topic"):
        parts.append(f"Topic: {item['topic']}")
    parts.append(f"Type: {item.get('type', 'theory')}")
    if item.get("content"):
        parts.append(item["content"])
    if item.get("question"):
        parts.append(f"Question: {item['question']}")
    if item.get("solution"):
        parts.append(f"Solution: {item['solution']}")
    if item.get("answer"):
        parts.append(f"Answer: {item['answer']}")
    return "\n".join(parts)


def embed_and_store(all_items: list[dict], doc_type: str) -> int:
    """Embed all items with Ollama and insert into Zilliz Cloud. Returns count inserted."""
    # Build embedding texts
    embed_texts = [build_embed_text(item) for item in all_items]

    # Generate embeddings via Ollama
    logger.info("Embedding %d items with %s...", len(embed_texts), settings.EMBED_MODEL)
    embeddings = []
    for i, text in enumerate(embed_texts):
        resp = ollama.embed(model=settings.EMBED_MODEL, input=text)
        embeddings.append(resp["embeddings"][0])
        if (i + 1) % 10 == 0:
            logger.info("Embedded %d/%d...", i + 1, len(embed_texts))

    embedding_dim = len(embeddings[0])
    logger.info("Embedding complete. Dimension: %d", embedding_dim)

    # Connect to Zilliz
    client = MilvusClient(uri=settings.ZILLIZ_URI, token=settings.ZILLIZ_TOKEN)

    # Create collection if it doesn't exist
    if not client.has_collection(settings.COLLECTION_NAME):
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            dimension=embedding_dim,
        )
        logger.info("Created collection '%s'", settings.COLLECTION_NAME)

    # Compute ID offset to avoid collisions with existing data
    try:
        stats = client.get_collection_stats(settings.COLLECTION_NAME)
        existing_count = int(stats.get("row_count", 0))
    except Exception:
        existing_count = 0

    id_offset = existing_count
    logger.info("Existing items: %d, new IDs start at %d", existing_count, id_offset)

    # Prepare data for insert
    data = []
    for i, item in enumerate(all_items):
        record = {
            "id": id_offset + i,
            "vector": embeddings[i],
            "item_id": item.get("id", f"item_{i}"),
            "type": item.get("type", "theory"),
            "subject": item.get("subject", ""),
            "chapter": item.get("chapter", ""),
            "topic": item.get("topic", ""),
            "content": item.get("content") or "",
            "question": item.get("question") or "",
            "options": json.dumps(item.get("options")) if item.get("options") else "",
            "answer": item.get("answer") or "",
            "solution": item.get("solution") or "",
            "images": json.dumps(item.get("images", [])),
            "page": item.get("page") or 0,
            "doc_type": doc_type,
        }
        data.append(record)

    client.insert(collection_name=settings.COLLECTION_NAME, data=data)
    logger.info("Inserted %d items into Zilliz collection '%s'",
                len(data), settings.COLLECTION_NAME)
    return len(data)