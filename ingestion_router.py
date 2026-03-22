import os
import uuid
import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from src.settings.config import settings
from src.services.ingestion_service import run_ingestion_pipeline

logger = logging.getLogger(__name__)

ingestion_router = APIRouter(tags=["Ingestion"])


@ingestion_router.post("/ingest")
async def ingest_pdf(
    file: UploadFile = File(..., description="PDF file to ingest"),
    doc_type: str = Form(default="book", description="Document type: book | mcq | handwritten"),
):
    """
    Upload a PDF and run the full ingestion pipeline:

    **PDF → OCR → LLM Structuring → Theory Splitting → Chapter Normalization → Embedding → Zilliz Cloud**

    This is a synchronous endpoint — it blocks until the full pipeline completes.
    Set a client timeout of at least 5 minutes for large PDFs.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    if doc_type not in ("book", "mcq", "handwritten"):
        raise HTTPException(
            status_code=400,
            detail="doc_type must be 'book', 'mcq', or 'handwritten'",
        )

    # Save uploaded file to disk
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    pdf_path = os.path.join(settings.UPLOAD_DIR, safe_name)

    content = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(content)

    logger.info("Saved uploaded PDF to %s (%d bytes)", pdf_path, len(content))

    try:
        result = run_ingestion_pipeline(pdf_path, doc_type)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.exception("Ingestion pipeline failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        # Clean up the uploaded PDF
        if os.path.exists(pdf_path):
            os.remove(pdf_path)