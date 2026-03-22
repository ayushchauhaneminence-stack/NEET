import logging
from fastapi import FastAPI
from ingestion_router import ingestion_router
from retrieval_router import retrieval_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="NEET2 Ingestion API", version="1.0.0")

app.include_router(ingestion_router)
app.include_router(retrieval_router)


@app.get("/health")
def health_check():
    return {"status": "ok"}
