import os
import logging
from collections import Counter

from src.services.ocr_service import extract_text_and_images
from src.services.structuring_service import structure_ocr_text
from src.services.embedding_service import embed_and_store

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(pdf_path: str, doc_type: str) -> dict:
    """
    Full pipeline: OCR -> LLM Structuring -> Embedding -> Zilliz.
    Returns a summary dict.
    """
    logger.info("Starting ingestion pipeline for %s (doc_type=%s)", pdf_path, doc_type)

    # Step 1: OCR extraction
    logger.info("Step 1/3: OCR extraction...")
    ocr_text, saved_images = extract_text_and_images(pdf_path)
    logger.info("OCR complete: %d chars, %d images", len(ocr_text), len(saved_images))

    # Step 2: LLM structuring (includes theory splitting + chapter normalization)
    logger.info("Step 2/3: LLM structuring...")
    all_items = structure_ocr_text(ocr_text)
    logger.info("Structuring complete: %d items", len(all_items))

    # Step 3: Embed + store in Zilliz
    logger.info("Step 3/3: Embedding and storing...")
    count = embed_and_store(all_items, doc_type)
    logger.info("Storage complete: %d items inserted", count)

    # Build summary
    type_counts = dict(Counter(item.get("type") for item in all_items))
    chapters = list(set(item.get("chapter", "") for item in all_items))

    return {
        "status": "success",
        "filename": os.path.basename(pdf_path),
        "doc_type": doc_type,
        "total_items": len(all_items),
        "type_counts": type_counts,
        "chapters": chapters,
        "images_extracted": len(saved_images),
        "ocr_text_length": len(ocr_text),
    }