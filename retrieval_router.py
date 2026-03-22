from fastapi import APIRouter, Query
from src.services.retrieval_service import retrieve

retrieval_router = APIRouter(tags=["Retrieval"])


@retrieval_router.get("/search")
async def search(q: str = Query(..., min_length=1, description="Search query")):
    """
    Full retrieval pipeline:
    - Classifies query (conceptual / derivation / numerical)
    - Semantic search in Zilliz vector DB
    - LLM re-ranking via DeepSeek
    - Confidence assessment with fallback
    - Grounded answer generation for conceptual/derivation queries
    """
    return retrieve(q)