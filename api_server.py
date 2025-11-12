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

# Global cache for LLM systems
llm_systems: Dict[str, LLMQuerySystem] = {}


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


def format_documents(documents) -> List[Dict[str, Any]]:
    """Format retrieved documents for API response"""
    formatted_docs = []
    for result, score in documents:
        payload = result['payload']
        formatted_docs.append({
            'file_path': payload.get('file_path', 'N/A'),
            'course': payload.get('course', 'N/A'),
            'file_type': payload.get('file_type', 'N/A'),
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
        "query_endpoint": "/api/v1/query"
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
