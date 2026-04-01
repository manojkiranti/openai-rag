from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.postgres import get_db
from app.db.qdrant import get_qdrant_client
from app.config import get_settings
from app.models.database import EmbeddedFile
from app.models.schemas import EmbedRequest, EmbedResponse, EmbeddedFileOut
from app.services.embedding import run_embed_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# In-memory store for background task status
_task_status: dict[str, dict] = {}


def _run_embed_task(task_id: str, filepaths: list[str]):
    _task_status[task_id] = {"status": "running"}
    try:
        result = run_embed_pipeline(filepaths or None)
        _task_status[task_id] = result
    except Exception as exc:
        logger.exception("Embed task %s failed: %s", task_id, exc)
        _task_status[task_id] = {"status": "failed", "error": str(exc)}


@router.post("/embed", response_model=EmbedResponse)
def embed_documents(
    request: EmbedRequest,
    background_tasks: BackgroundTasks,
):
    """Trigger document embedding as a background task."""
    task_id = str(uuid.uuid4())
    _task_status[task_id] = {"status": "queued"}
    background_tasks.add_task(_run_embed_task, task_id, request.filepaths)
    return EmbedResponse(
        message="Embedding started in background",
        task_id=task_id,
    )


@router.get("/embed/status/{task_id}")
def embed_status(task_id: str):
    """Check the status of an embedding task."""
    status = _task_status.get(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@router.get("", response_model=list[EmbeddedFileOut])
def list_documents(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List all embedded documents."""
    rows = (
        db.query(EmbeddedFile)
        .order_by(EmbeddedFile.processed_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return rows


@router.get("/{doc_id}", response_model=EmbeddedFileOut)
def get_document(doc_id: int, db: Session = Depends(get_db)):
    """Get a single embedded document by ID."""
    doc = db.query(EmbeddedFile).filter_by(id=doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    """
    Remove a document record from Postgres and its vectors from Qdrant.
    """
    doc = db.query(EmbeddedFile).filter_by(id=doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove vectors from Qdrant that match this file
    settings = get_settings()
    client = get_qdrant_client()
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=doc.filepath))]
            ),
        )
        logger.info("Deleted Qdrant vectors for %s", doc.filepath)
    except Exception as exc:
        logger.warning("Failed to delete Qdrant vectors for %s: %s", doc.filepath, exc)

    # Remove from Postgres
    db.delete(doc)
    db.commit()

    return {"message": f"Deleted document: {doc.filename}", "filepath": doc.filepath}
