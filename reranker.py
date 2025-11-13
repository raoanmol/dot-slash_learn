'''
Reranker Module for RAG Results

Uses Qwen3-Reranker-4B to rerank documents retrieved from vector search.
This provides more accurate relevance scoring by considering the full context
of both the query and document together.

This is meant to be imported and used as a module in other scripts.

Usage:
    from reranker import DocumentReranker

    reranker = DocumentReranker()

    # results should be a list of dicts with 'payload' key
    reranked_results = reranker.rerank(query, results)

    # Returns list of tuples: (result, rerank_score), sorted by score descending
    for result, score in reranked_results:
        print(f"Score: {score:.4f}")
        print(f"File: {result['payload']['file_path']}")
'''

import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class DocumentReranker:
    """
    Reranker that uses Qwen3-Reranker-4B to rerank search results.
    """

    def __init__(self, device: str = None, verbose: bool = True):
        """
        Initialize the reranker model.

        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
            verbose: Whether to print loading messages.
        """
        self.verbose = verbose

        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.verbose:
            print('Loading Qwen/Qwen3-Reranker-4B model...')

        # Load reranker model with sequence classification head
        self.rerank_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Reranker-4B')
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained('Qwen/Qwen3-Reranker-4B')
        self.rerank_model.to(self.device)
        self.rerank_model.eval()

        if self.verbose:
            print(f'Reranker loaded on {self.device}\n')

    def compute_rerank_score(self, query: str, document: str) -> float:
        """
        Compute reranking score for a query-document pair.
        Higher scores indicate better relevance.

        Args:
            query: The search query
            document: The document text to score

        Returns:
            Relevance score (higher = more relevant)
        """
        # Truncate document to prevent OOM (keep first 3000 chars, ~750 tokens)
        if len(document) > 3000:
            document = document[:3000] + "..."

        inputs = self.rerank_tokenizer(
            query,
            document,
            return_tensors='pt',
            truncation=True,
            max_length=4096,  # Reduced from 8192 to save memory
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.rerank_model(**inputs)

            # For sequence classification models, logits contain the relevance scores
            # Typically: [batch_size, num_classes] where num_classes might be 1 or 2
            logits = outputs.logits

            # If binary classification (2 classes), use the positive class score
            if logits.shape[-1] == 2:
                # Class 1 is typically "relevant"
                relevance_score = logits[0, 1].cpu().item()
            else:
                # Single output score
                relevance_score = logits[0, 0].cpu().item()

        # Clear CUDA cache after each scoring to prevent memory accumulation
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return relevance_score

    def rerank(self, query: str, results: List[Dict[str, Any]], min_score: float = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank search results using the reranker model.

        Args:
            query: The search query
            results: List of search results. Each result should be a dict with either:
                     - 'payload' key containing document metadata, OR
                     - document metadata directly in the dict
            min_score: Minimum reranker score threshold. Documents below this score are filtered out.

        Returns:
            List of tuples: (result, rerank_score), sorted by rerank_score descending
        """
        if self.verbose:
            print(f'Reranking {len(results)} results...')

        reranked = []

        for idx, result in enumerate(results):
            # Handle both formats: result might have 'payload' key or be the payload itself
            if 'payload' in result:
                payload = result['payload']
            else:
                payload = result

            # Build document context from metadata and content
            doc_context = f"Course: {payload.get('course', '')}\n"
            doc_context += f"Type: {payload.get('file_type', '')}\n"

            if payload.get('title'):
                doc_context += f"Title: {payload.get('title', '')}\n"
            if payload.get('description'):
                doc_context += f"Description: {payload.get('description', '')}\n"

            document_text = payload.get('content_preview', '')
            doc_context += f"\nContent:\n{document_text}"

            # Compute reranking score
            rerank_score = self.compute_rerank_score(query, doc_context)

            reranked.append((result, rerank_score))

            if self.verbose:
                print(f'  [{idx+1}/{len(results)}] Score: {rerank_score:.4f}')

        # Sort by rerank score (descending)
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Filter by minimum score if specified
        if min_score is not None:
            original_count = len(reranked)
            reranked = [(result, score) for result, score in reranked if score >= min_score]
            if self.verbose and original_count > len(reranked):
                print(f'  Filtered out {original_count - len(reranked)} documents below threshold {min_score:.1f}')

        if self.verbose:
            print()

        return reranked

    def offload_to_cpu(self):
        """Move the reranker model to CPU to free VRAM"""
        if self.device.type == 'cuda':
            self.rerank_model.to('cpu')
            if self.verbose:
                print('Reranker moved to CPU')
