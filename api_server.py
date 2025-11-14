'''
FastAPI REST API Server for RAG-LLM Query System

Simple API for querying course materials with LLM assistance.
Assumes Qdrant is already set up with embedded vectors.

Usage:
    uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
'''

import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from llm import LLMQuerySystem
from course_embedder import CourseEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG-LLM Query API",
    description="Query course materials using RAG with LLM assistance",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for LLM systems and embedders
llm_systems: Dict[str, LLMQuerySystem] = {}
embedders: Dict[str, CourseEmbedder] = {}


# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="The question to ask", min_length=1)
    collection_name: str = Field(
        default="cs_materials",
        description="Qdrant collection to search"
    )
    show_context: bool = Field(
        default=False,
        description="Include retrieved documents in response"
    )
    max_length: int = Field(
        default=2048,
        description="Maximum length of generated response",
        ge=100,
        le=4096
    )
    enable_guardrails: bool = Field(
        default=True,
        description="Enable safety guardrails"
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    context: Optional[List[Dict[str, Any]]] = None
    success: bool
    error: Optional[str] = None


class EmbedRequest(BaseModel):
    """Request model for embed endpoint"""
    file_paths: List[str] = Field(
        ...,
        description="List of file paths to embed (relative to downloaded_materials by default)",
        min_length=1
    )
    learning_objective: str = Field(
        ...,
        description="Description of what these files teach",
        min_length=1
    )
    collection_name: str = Field(
        default="cs_materials",
        description="Qdrant collection to store embeddings"
    )
    course: Optional[str] = Field(
        default=None,
        description="Optional course identifier (e.g., '6_0001')"
    )
    group_type: Optional[str] = Field(
        default=None,
        description="Optional group type (e.g., 'lecture', 'assignment')"
    )
    group_id: Optional[str] = Field(
        default=None,
        description="Optional group identifier"
    )
    absolute_paths: bool = Field(
        default=False,
        description="Treat provided file paths as absolute instead of resolving against the default materials directory"
    )
    module_name: Optional[str] = Field(
        default=None,
        description="Optional friendly name for the module or grouping"
    )
    module_item_ids: Optional[List[str]] = Field(
        default=None,
        description="Canvas module item identifiers included in this embed request"
    )
    file_metadata: Optional[List[Dict[str, Optional[str]]]] = Field(
        default=None,
        description="Per-file metadata including module_item_id references"
    )
    ingested_at: Optional[str] = Field(
        default=None,
        description="Timestamp for when this embed request was initiated"
    )


class EmbedResponse(BaseModel):
    """Response model for embed endpoint"""
    success: bool
    total_chunks: int
    files_processed: int
    skills: List[str] = []
    errors: List[str] = []
    message: Optional[str] = None

    class FileMetadataEntry(BaseModel):
        """Metadata describing the relationship between a file and Canvas content."""

        file_path: str = Field(
            ...,
            description="File path as provided in the embed request"
        )
        module_item_id: Optional[str] = Field(
            default=None,
            description="Canvas module item identifier associated with this file"
        )
        requested_path: Optional[str] = Field(
            default=None,
            description="Original path string supplied by the caller"
        )


class DeleteFileRequest(BaseModel):
    """Request model for delete file endpoint"""
    file_paths: List[str] = Field(
        ...,
        description="List of exact file paths to delete",
        min_length=1
    )
    collection_name: str = Field(
        default="cs_materials",
        description="Qdrant collection to delete from"
    )


class DeleteFileResponse(BaseModel):
    """Response model for delete file endpoint"""
    success: bool
    chunks_deleted: int
    files_deleted: int
    message: str
    errors: List[str] = []
    error: Optional[str] = None


class ModuleItemStatus(BaseModel):
    """Status details for a specific Canvas module item."""

    module_item_id: Optional[str]
    file_paths: List[str] = []
    absolute_file_paths: List[str] = []
    file_count: int = 0
    chunk_count: int = 0
    skills: List[str] = []


class ModuleStatus(BaseModel):
    """Aggregated status for an embedded Canvas module."""

    group_id: str
    module_name: Optional[str] = None
    learning_objective: Optional[str] = None
    file_paths: List[str] = []
    absolute_file_paths: List[str] = []
    file_count: int = 0
    chunk_count: int = 0
    skills: List[str] = []
    module_item_ids: List[str] = []
    items: List[ModuleItemStatus] = []
    ingested_at: Optional[str] = None


class ModuleStatusResponse(BaseModel):
    """Response model describing existing embedded materials."""

    success: bool
    collection_name: str
    course: Optional[str]
    group_type: Optional[str]
    modules: List[ModuleStatus] = []


# ============================================================================
# Helper Functions
# ============================================================================

def get_llm_system(
    collection_name: str,
    enable_guardrails: bool = True
) -> LLMQuerySystem:
    """Get or create an LLM system (with caching)"""
    cache_key = f"{collection_name}_{enable_guardrails}"

    if cache_key not in llm_systems:
        logger.info(f"Initializing LLM system for collection: {collection_name}")
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

        llm_systems[cache_key] = LLMQuerySystem(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            enable_guardrails=enable_guardrails
        )

    return llm_systems[cache_key]


def get_embedder(collection_name: str) -> CourseEmbedder:
    """Get or create a CourseEmbedder (with caching)"""
    if collection_name not in embedders:
        logger.info(f"Initializing CourseEmbedder for collection: {collection_name}")
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

        embedders[collection_name] = CourseEmbedder(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            collection_name=collection_name
        )

    return embedders[collection_name]


def format_documents(documents) -> List[Dict[str, Any]]:
    """Format retrieved documents for API response"""
    formatted_docs = []
    for result, score in documents:
        payload = result['payload']
        formatted_docs.append({
            'file_path': payload.get('file_path', 'N/A'),
            'course': payload.get('course', 'N/A'),
            'file_type': payload.get('file_type', 'N/A'),
            'skills': payload.get('skills', []),
            'relevance_score': score,
            'content_preview': payload.get('content_preview', ''),
            'chunk_index': payload.get('chunk_index'),
            'total_chunks': payload.get('total_chunks')
        })
    return formatted_docs


def get_materials_root() -> Path:
    """Return the base directory for downloaded course materials."""
    override = os.getenv("MATERIALS_ROOT")
    if override:
        base = Path(override).expanduser()
    else:
        base = (Path(__file__).resolve().parent.parent / "downloaded_materials").expanduser()
    try:
        return base.resolve(strict=False)
    except Exception:
        return base


def resolve_request_paths(
    file_paths: List[str],
    *,
    absolute_paths: bool = False,
) -> Tuple[List[Tuple[str, Path]], List[str], Path]:
    """Resolve incoming file path strings to concrete Paths.

    Returns a tuple of (resolved_pairs, warnings, materials_root) where resolved_pairs
    contains (original, resolved) path tuples.
    """
    materials_root = get_materials_root()
    try:
        materials_root_resolved = materials_root.resolve(strict=False)
    except Exception:
        materials_root_resolved = materials_root

    resolved_pairs: List[Tuple[str, Path]] = []
    warnings: List[str] = []

    for original in file_paths:
        if not original:
            warnings.append("Empty file path provided in request.")
            continue

        candidate = Path(original).expanduser()

        if not (absolute_paths or candidate.is_absolute()):
            candidate = (materials_root_resolved / candidate).expanduser()

        try:
            resolved = candidate.resolve(strict=False)
        except Exception:
            resolved = candidate

        if not (absolute_paths or Path(original).is_absolute()):
            try:
                resolved.relative_to(materials_root_resolved)
            except ValueError:
                warnings.append(
                    f"Resolved path '{resolved}' is outside of materials root '{materials_root_resolved}'."
                )
                continue

        if not resolved.exists():
            warnings.append(f"File not found: {resolved}")

        resolved_pairs.append((original, resolved))

    return resolved_pairs, warnings, materials_root_resolved


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "RAG-LLM Query API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "query": "/api/v1/query",
            "embed": "/api/v1/embed",
            "delete": "/api/v1/embed (DELETE)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check"""
    try:
        from qdrant_client import QdrantClient

        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))

        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        collections = client.get_collections()

        return {
            "status": "healthy",
            "qdrant_connected": True,
            "collections": [col.name for col in collections.collections]
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "qdrant_connected": False,
            "error": str(e)
        }


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_llm(request: QueryRequest):
    """
    Query the LLM with RAG context from course materials.

    Pipeline:
    1. Input validation (guardrails if enabled)
    2. Two-stage RAG retrieval with reranking
    3. LLM response generation
    4. Output validation (guardrails if enabled)
    """
    try:
        logger.info(f"Query: {request.query[:100]}... | Collection: {request.collection_name}")

        # Get LLM system
        system = get_llm_system(
            collection_name=request.collection_name,
            enable_guardrails=request.enable_guardrails
        )

        # Perform query
        response = system.query(
            user_query=request.query,
            show_context=False,
            max_length=request.max_length
        )

        result = QueryResponse(
            answer=response,
            success=True
        )

        # Optionally include context
        if request.show_context:
            documents = system.query_engine.search(
                query=request.query,
                top_k=3,
                use_reranker=True,
                rerank_candidates=50,
                stage1_top_k=7,
                verbose=False
            )
            result.context = format_documents(documents)

        logger.info(f"Query successful")
        return result

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return QueryResponse(
            answer="",
            success=False,
            error=str(e)
        )


@app.post("/api/v1/embed", response_model=EmbedResponse)
async def embed_files(request: EmbedRequest):
    """
    Embed files into a Qdrant collection with skill-based classification.

    Pipeline:
    1. Extract text from all files
    2. Use LLM to extract 5-6 high-level skills from combined content
    3. Chunk documents
    4. Parallel processing:
       - Classify each chunk into skills (multithreaded)
       - Generate embeddings for each chunk
    5. Store in Qdrant with skill metadata
    """
    try:
        logger.info(f"Embedding {len(request.file_paths)} files into collection: {request.collection_name}")
        logger.info(f"Learning objective: {request.learning_objective[:100]}...")

        resolved_pairs, resolution_warnings, materials_root = resolve_request_paths(
            request.file_paths,
            absolute_paths=request.absolute_paths,
        )
        resolved_paths = [str(resolved) for _, resolved in resolved_pairs]

        metadata_by_original: Dict[str, Any] = {}
        if request.file_metadata:
            for entry in request.file_metadata:
                if isinstance(entry, dict):
                    file_path_value = entry.get("file_path")
                    if not file_path_value:
                        continue
                    try:
                        parsed_entry = FileMetadataEntry(**entry)
                    except Exception:
                        logger.warning("Skipping invalid file metadata entry: %s", entry)
                        continue
                    metadata_by_original[str(file_path_value)] = parsed_entry
                else:
                    metadata_by_original[entry.file_path] = entry

        file_metadata_map: Dict[str, Dict[str, Any]] = {}
        for original, resolved in resolved_pairs:
            metadata_entry = metadata_by_original.get(original)
            requested_path = original
            payload_metadata: Dict[str, Any] = {
                "requested_path": metadata_entry.requested_path if metadata_entry else requested_path,
            }
            if metadata_entry and metadata_entry.module_item_id is not None:
                payload_metadata["module_item_id"] = metadata_entry.module_item_id
            file_metadata_map[str(resolved)] = payload_metadata

        if resolution_warnings:
            for warning in resolution_warnings:
                logger.warning("Embed request warning: %s", warning)

        if not resolved_paths:
            message = "No valid file paths could be resolved for embedding."
            return EmbedResponse(
                success=False,
                total_chunks=0,
                files_processed=0,
                skills=[],
                errors=resolution_warnings or [message],
                message=message,
            )

        logger.info("Resolved materials root: %s", materials_root)
        logger.info("Resolved file paths: %s", resolved_paths)

        # Get embedder (creates collection if needed)
        embedder = get_embedder(request.collection_name)

        # Embed files
        ingested_at = request.ingested_at or datetime.now(timezone.utc).isoformat()
        result = embedder.embed_files(
            file_paths=resolved_paths,
            learning_objective=request.learning_objective,
            course=request.course,
            group_type=request.group_type,
            group_id=request.group_id,
            module_name=request.module_name,
            module_item_ids=request.module_item_ids,
            file_metadata=file_metadata_map,
            ingested_at=ingested_at,
        )

        combined_errors = resolution_warnings + result['errors']

        response = EmbedResponse(
            success=result['success'],
            total_chunks=result['total_chunks'],
            files_processed=result['files_processed'],
            skills=result.get('skills', []),
            errors=combined_errors,
            message=f"Successfully embedded {result['files_processed']} files into {result['total_chunks']} chunks with {len(result.get('skills', []))} skills"
        )

        logger.info(f"Embedding complete: {result['total_chunks']} chunks created with skills: {result.get('skills', [])}")
        return response

    except Exception as e:
        logger.error(f"Embedding failed: {e}", exc_info=True)
        return EmbedResponse(
            success=False,
            total_chunks=0,
            files_processed=0,
            skills=[],
            errors=[str(e)],
            message="Embedding failed"
        )


@app.get("/api/v1/embed/status", response_model=ModuleStatusResponse)
async def embed_status(
    collection_name: str = Query(default="cs_materials"),
    course: Optional[str] = Query(default=None),
    group_type: Optional[str] = Query(default="module"),
):
    """Return aggregated knowledge-base status for modules."""
    try:
        embedder = get_embedder(collection_name)
        modules = embedder.collect_module_status(course=course, group_type=group_type)
        return ModuleStatusResponse(
            success=True,
            collection_name=collection_name,
            course=course,
            group_type=group_type,
            modules=modules,
        )
    except Exception as exc:
        logger.error("Failed to load embed status: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Unable to fetch embed status.") from exc


@app.delete("/api/v1/embed", response_model=DeleteFileResponse)
async def delete_files(request: DeleteFileRequest):
    """
    Delete all chunks associated with specific files from a collection.

    This removes all embedded chunks that were created from the specified files.
    Accepts a list of file paths and deletes them all.
    """
    try:
        logger.info(f"Deleting {len(request.file_paths)} files from collection: {request.collection_name}")

        resolved_pairs, resolution_warnings, materials_root = resolve_request_paths(
            request.file_paths,
            absolute_paths=False,
        )

        if resolution_warnings:
            for warning in resolution_warnings:
                logger.warning("Delete request warning: %s", warning)

        if not resolved_pairs:
            message = "No valid file paths could be resolved for deletion."
            return DeleteFileResponse(
                success=False,
                chunks_deleted=0,
                files_deleted=0,
                message=message,
                errors=resolution_warnings,
                error=message,
            )

        logger.info("Resolved materials root for deletion: %s", materials_root)

        # Get embedder
        embedder = get_embedder(request.collection_name)

        # Delete files
        total_chunks_deleted = 0
        files_deleted = 0
        errors = list(resolution_warnings)

        for original_path, resolved_path in resolved_pairs:
            logger.info("  Deleting: %s (requested: %s)", resolved_path, original_path)
            result = embedder.delete_file(str(resolved_path))

            if result['success']:
                total_chunks_deleted += result['chunks_deleted']
                if result['chunks_deleted'] > 0:
                    files_deleted += 1
                else:
                    errors.append(f"No chunks found for: {original_path}")
            else:
                errors.append(
                    f"Failed to delete {original_path}: {result.get('error', 'Unknown error')}"
                )

        success = files_deleted > 0 or (len(errors) == 0 and len(request.file_paths) > 0)

        response = DeleteFileResponse(
            success=success,
            chunks_deleted=total_chunks_deleted,
            files_deleted=files_deleted,
            message=f"Successfully deleted {files_deleted} files ({total_chunks_deleted} chunks)",
            errors=errors
        )

        logger.info(f"Deletion complete: {files_deleted} files, {total_chunks_deleted} chunks removed")
        return response

    except Exception as e:
        logger.error(f"Deletion failed: {e}", exc_info=True)
        return DeleteFileResponse(
            success=False,
            chunks_deleted=0,
            files_deleted=0,
            message="Deletion failed",
            errors=[],
            error=str(e)
        )


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting RAG-LLM API Server...")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = os.getenv("QDRANT_PORT", "6333")
    logger.info(f"Qdrant: {qdrant_host}:{qdrant_port}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")
    global llm_systems
    llm_systems.clear()


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
        log_level="info"
    )
