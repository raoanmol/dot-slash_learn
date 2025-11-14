'''
FastAPI REST API Server for RAG-LLM Query System

Simple API for querying course materials with LLM assistance.
Assumes Qdrant is already set up with embedded vectors.

Usage:
    uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
'''

import os
import logging
from typing import Optional, List, Dict, Any

from fastapi import FastAPI
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
        description="List of absolute file paths to embed",
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


class EmbedResponse(BaseModel):
    """Response model for embed endpoint"""
    success: bool
    total_chunks: int
    files_processed: int
    skills: List[str] = []
    errors: List[str] = []
    message: Optional[str] = None


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
            collection_name=collection_name,
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

        # Get embedder (creates collection if needed)
        embedder = get_embedder(request.collection_name)

        # Embed files
        result = embedder.embed_files(
            file_paths=request.file_paths,
            learning_objective=request.learning_objective,
            course=request.course,
            group_type=request.group_type,
            group_id=request.group_id
        )

        response = EmbedResponse(
            success=result['success'],
            total_chunks=result['total_chunks'],
            files_processed=result['files_processed'],
            skills=result.get('skills', []),
            errors=result['errors'],
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


@app.delete("/api/v1/embed", response_model=DeleteFileResponse)
async def delete_files(request: DeleteFileRequest):
    """
    Delete all chunks associated with specific files from a collection.

    This removes all embedded chunks that were created from the specified files.
    Accepts a list of file paths and deletes them all.
    """
    try:
        logger.info(f"Deleting {len(request.file_paths)} files from collection: {request.collection_name}")

        # Get embedder
        embedder = get_embedder(request.collection_name)

        # Delete files
        total_chunks_deleted = 0
        files_deleted = 0
        errors = []

        for file_path in request.file_paths:
            logger.info(f"  Deleting: {file_path}")
            result = embedder.delete_file(file_path)

            if result['success']:
                total_chunks_deleted += result['chunks_deleted']
                if result['chunks_deleted'] > 0:
                    files_deleted += 1
                else:
                    errors.append(f"No chunks found for: {file_path}")
            else:
                errors.append(f"Failed to delete {file_path}: {result.get('error', 'Unknown error')}")

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
