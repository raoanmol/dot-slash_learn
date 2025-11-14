'''
RAG Query System with Two-Stage Reranking

Advanced RAG implementation with sibling file expansion:
1. Takes a user query
2. Embeds it using Qwen/Qwen3-Embedding-4B
3. Searches Qdrant for top 10 relevant course materials
4. STAGE 1: Reranks using Qwen/Qwen3-Reranker-4B to get top 3
5. STAGE 2: Expands top 3 with their sibling files (from Qdrant or disk)
6. Final rerank of expanded set to get final top 3 documents

This module is designed to be imported by llm.py, not run standalone.

Usage:
    from query import QueryEngine

    engine = QueryEngine(collection_name='cs_materials')
    results = engine.search(query, top_k=3, use_reranker=True)
'''

import argparse
import torch
import os
import PyPDF2
from typing import List
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel


class QueryEngine:

    def __init__(self, qdrant_host: str = 'localhost', qdrant_port: int = 6333, collection_name: str = 'cs_materials'):
        self.collection_name = collection_name

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Load embedding model (same as course_embedder.py)
        print('Loading Qwen/Qwen3-Embedding-4B model...')
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B')
        self.model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B')
        self.model.to(self.device)
        self.model.eval()

        # Connect to Qdrant
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)

        # Verify collection exists
        try:
            self.client.get_collection(collection_name)
            print(f'Connected to collection: {collection_name}\n')
        except Exception as e:
            print(f'Error: Collection "{collection_name}" not found!')
            print(f'Available collections:')
            collections = self.client.get_collections()
            for col in collections.collections:
                print(f'  - {col.name}')
            raise e

    def embed_query(self, query: str) -> List[float]:
        """Embed the query using the same model as course materials"""
        inputs = self.tokenizer(
            query,
            return_tensors='pt',
            truncation=True,
            max_length=8192,
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()

        return embedding.cpu().numpy().tolist()

    def search(self, query: str, top_k: int = 3, use_reranker: bool = False, rerank_candidates: int = 50, stage1_top_k: int = 7, min_score: float = 7.0, verbose: bool = True):
        """
        Search for relevant documents with two-stage reranking.

        Pipeline:
        1. Retrieve top 50 candidates from vector search
        2. STAGE 1: Rerank to get top 7
        3. Expand top 7 with neighboring chunks
        4. STAGE 2: Rerank expanded set to get final top_k (default: 3)
        5. Filter out documents with reranker score < min_score

        Args:
            top_k: Final number of documents to return (default: 3)
            rerank_candidates: Number of initial candidates to retrieve (default: 50)
            stage1_top_k: Number of documents to keep after stage 1 rerank (default: 7)
            min_score: Minimum reranker score threshold (default: 7.0). Documents below this are filtered out.

        Returns:
            List of tuples: (result_dict, rerank_score)
        """
        if verbose:
            print(f'Query: "{query}"\n')
            print('='*80)
            print('Embedding query...')

        query_vector = self.embed_query(query)

        # Determine how many candidates to retrieve
        num_to_retrieve = rerank_candidates if use_reranker else top_k

        if verbose:
            print(f'Searching for top {num_to_retrieve} results...\n')

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=num_to_retrieve
        )

        if not results:
            if verbose:
                print('No results found!')
            return []

        # Convert results to the format expected by reranker
        results_list = []
        for hit in results:
            results_list.append({
                'payload': hit.payload,
                'score': hit.score,
                'id': hit.id
            })

        # Optionally rerank results
        if use_reranker:
            from reranker import DocumentReranker

            reranker = DocumentReranker(verbose=verbose)

            # STAGE 1: First rerank to get top N (default: 7)
            if verbose:
                print('='*80)
                print('STAGE 1: Initial Reranking')
                print('='*80)
            first_rerank = reranker.rerank(query, results_list)
            top_n_initial = first_rerank[:stage1_top_k]

            # STAGE 2: Expand with neighboring chunks
            if verbose:
                print('='*80)
                print('STAGE 2: Expanding with Neighboring Chunks')
                print('='*80)
            expanded_results = self._expand_with_siblings(top_n_initial, results_list, verbose=verbose)

            if verbose:
                print(f'\nExpanded from {len(top_n_initial)} to {len(expanded_results)} chunks (including neighbors)')
                print('\n' + '='*80)
                print('STAGE 2: Final Reranking')
                print('='*80)
            final_rerank = reranker.rerank(query, expanded_results, min_score=min_score)

            # Offload reranker to CPU to free VRAM
            reranker.offload_to_cpu()
            if verbose:
                print()

            # Take top_k after filtering by min_score
            # Note: reranker already filters by min_score, so we just need to limit to top_k
            results_to_display = final_rerank[:top_k]

            if verbose and len(final_rerank) == 0:
                print(f'⚠ No documents passed the minimum score threshold of {min_score:.1f}')
            elif verbose and len(final_rerank) < top_k:
                print(f'⚠ Only {len(final_rerank)} documents passed the minimum score threshold of {min_score:.1f}')

            return results_to_display
        else:
            # Return results without reranking
            return [(r, r['score']) for r in results_list[:top_k]]

    def _expand_with_siblings(self, top_results, all_results, verbose: bool = True):
        """
        Expand top results with their sibling files/chunks from Qdrant.

        For chunked documents: Expands with neighboring chunks from same file
        For non-chunked documents: Expands with sibling files from same group

        Args:
            top_results: List of (result, score) tuples from first rerank
            all_results: Original list of all retrieved results
            verbose: Whether to print progress messages

        Returns:
            List of result dicts (without scores) ready for reranking
        """
        expanded = []
        seen_ids = set()

        # First, add the top 3 results themselves
        for result, score in top_results:
            result_id = result.get('id')
            if result_id and result_id not in seen_ids:
                expanded.append(result)
                seen_ids.add(result_id)

        # Then, expand based on whether documents are chunked
        for result, score in top_results:
            payload = result['payload']

            # Check if this is a chunk
            is_chunked = payload.get('is_chunked', False)

            if is_chunked:
                # Expand with neighboring chunks from same document
                file_path = payload.get('file_path')
                chunk_index = payload.get('chunk_index', 0)
                total_chunks = payload.get('total_chunks', 1)

                if verbose:
                    print(f"\nExpanding chunk {chunk_index+1}/{total_chunks} from: {file_path.split('/')[-1]}")

                # Find prev and next chunks
                for offset in [-1, 1]:  # Previous and next chunk
                    neighbor_idx = chunk_index + offset
                    if 0 <= neighbor_idx < total_chunks:
                        neighbor = self._find_chunk_by_index(file_path, neighbor_idx, verbose=verbose)
                        if neighbor and neighbor.get('id') not in seen_ids:
                            expanded.append(neighbor)
                            seen_ids.add(neighbor.get('id'))
                            if verbose:
                                print(f"  ✓ Added chunk {neighbor_idx+1}/{total_chunks}")

            # Commented out: Original sibling file expansion behavior
            # This was used before chunking to expand with related files from the same group
            # With chunking, we focus on neighboring chunks instead for better precision
            #
            # else:
            #     # Original behavior: expand with sibling files
            #     sibling_paths = payload.get('sibling_files', [])
            #
            #     if sibling_paths:
            #         if verbose:
            #             print(f"\nFinding siblings for: {payload.get('file_path', 'N/A').split('/')[-1]}")
            #             print(f"  Looking for {len(sibling_paths)} sibling file(s)...")
            #
            #         found_count = 0
            #         for sibling_path in sibling_paths:
            #             # Try to find this sibling in our original results
            #             sibling_result = self._find_document_by_path(sibling_path, verbose=verbose)
            #
            #             if sibling_result and sibling_result.get('id') not in seen_ids:
            #                 expanded.append(sibling_result)
            #                 seen_ids.add(sibling_result.get('id'))
            #                 found_count += 1
            #
            #         if verbose:
            #             print(f"  Found {found_count} sibling(s) in Qdrant")

        return expanded

    def _find_chunk_by_index(self, file_path: str, chunk_index: int, verbose: bool = True):
        """
        Find a specific chunk from a document by its file path and chunk index.

        Args:
            file_path: The file path of the document
            chunk_index: The index of the chunk to find
            verbose: Whether to print messages

        Returns:
            Result dict if found, None otherwise
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=file_path)
                        ),
                        FieldCondition(
                            key="chunk_index",
                            match=MatchValue(value=chunk_index)
                        )
                    ]
                ),
                limit=1
            )

            if results[0]:  # results is a tuple (points, next_page_offset)
                hit = results[0][0]  # Get first point
                return {
                    'payload': hit.payload,
                    'score': 0.0,  # No score from scroll
                    'id': hit.id
                }
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not find chunk {chunk_index}: {e}")
            return None

    def _find_document_by_path(self, file_path, verbose: bool = True):
        """
        Find a document in Qdrant by its file path, or read from disk if not found.

        Args:
            file_path: The file path to search for
            verbose: Whether to print messages

        Returns:
            Result dict if found, None otherwise
        """
        # First, try to find in Qdrant
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=file_path)
                        )
                    ]
                ),
                limit=1
            )

            if results[0]:  # results is a tuple (points, next_page_offset)
                hit = results[0][0]  # Get first point
                return {
                    'payload': hit.payload,
                    'score': 0.0,  # No score from scroll
                    'id': hit.id
                }
        except Exception as e:
            pass  # Fall through to reading from disk

        # If not in Qdrant, try to read from disk
        return self._load_document_from_disk(file_path, verbose=verbose)

    def _load_document_from_disk(self, file_path, verbose: bool = True):
        """
        Load a document from disk and create a pseudo-document for reranking.

        Args:
            file_path: The file path to load
            verbose: Whether to print messages

        Returns:
            Result dict with extracted content, or None if file cannot be read
        """
        if not os.path.exists(file_path):
            if verbose:
                print(f"  Warning: File not found on disk: {file_path.split('/')[-1]}")
            return None

        try:
            # Extract text based on file type
            text_content = self._extract_text_from_file(file_path)

            if not text_content or not text_content.strip():
                if verbose:
                    print(f"  Warning: No content extracted from {file_path.split('/')[-1]}")
                return None

            # Get file extension and type
            file_ext = os.path.splitext(file_path)[1]
            file_type = self._determine_file_type(file_ext)

            # Create a pseudo-document with the same structure as Qdrant documents
            # Limit content_preview to prevent OOM during reranking
            pseudo_doc = {
                'payload': {
                    'file_path': file_path,
                    'file_type': file_type,
                    'course': 'N/A',  # We don't have this info for on-the-fly docs
                    'content_preview': text_content[:3000],  # Reduced to save memory during reranking
                    'content_length': len(text_content),
                    'loaded_from_disk': True,  # Flag to indicate this was loaded on-the-fly
                    'full_content': text_content[:5000]  # Limit full content too
                },
                'score': 0.0,
                'id': f'disk_{hash(file_path)}'  # Generate a pseudo-ID
            }

            if verbose:
                print(f"  ✓ Loaded from disk: {file_path.split('/')[-1]} ({len(text_content)} chars)")
            return pseudo_doc

        except Exception as e:
            if verbose:
                print(f"  Warning: Could not read {file_path.split('/')[-1]}: {e}")
            return None

    def _extract_text_from_file(self, file_path):
        """Extract text from various file types"""
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.pdf':
            return self._extract_text_from_pdf(file_path)
        elif file_ext in ['.py', '.txt', '.md', '.json', '.js', '.java', '.cpp', '.c', '.h']:
            return self._extract_text_from_code(file_path)
        else:
            # Try to read as plain text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                return ''

    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + '\n'
                return text.strip()
        except Exception as e:
            print(f"  Warning: Error extracting PDF: {e}")
            return ''

    def _extract_text_from_code(self, code_path):
        """Extract text from code/text files"""
        try:
            with open(code_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"  Warning: Error reading code file: {e}")
            return ''

    def _determine_file_type(self, extension):
        """Determine file type from extension"""
        ext_map = {
            '.pdf': 'reference',
            '.py': 'code',
            '.js': 'code',
            '.java': 'code',
            '.cpp': 'code',
            '.c': 'code',
            '.h': 'code',
            '.txt': 'text',
            '.md': 'documentation',
            '.json': 'data'
        }
        return ext_map.get(extension.lower(), 'unknown')

    def _print_results(self, results_to_display, reranked: bool = False, original_results=None):
        """Helper method to print search results"""
        score_label = 'Reranker Score' if reranked else 'Score'

        print(f'Found {len(results_to_display)} results:\n')
        print('='*80)

        for idx, (result, score) in enumerate(results_to_display, 1):
            payload = result['payload']

            print(f'\n[Result {idx}] ({score_label}: {score:.4f})', end='')

            # If reranked, also show original embedding score
            if reranked:
                original_score = result.get('score', 0.0)
                print(f' (Embedding: {original_score:.4f})', end='')

            print()
            print('-'*80)

            # Basic info
            print(f'Course: {payload.get("course", "N/A")}')
            print(f'File: {payload.get("file_path", "N/A")}')
            print(f'Type: {payload.get("file_type", "N/A")}')

            # Group info
            group_type = payload.get('group_type', 'N/A')
            print(f'Group Type: {group_type}')

            if group_type == 'lecture':
                lecture_num = payload.get('lecture_number', 'N/A')
                title = payload.get('title', 'N/A')
                print(f'Lecture {lecture_num}: {title}')
            elif group_type == 'assignment':
                assign_num = payload.get('assignment_number', 'N/A')
                title = payload.get('title', 'N/A')
                print(f'Assignment {assign_num}: {title}')

            # Description
            description = payload.get('description', '')
            if description:
                print(f'Description: {description}')

            # Content preview
            preview = payload.get('content_preview', '')
            if preview:
                print(f'\nContent Preview:')
                print(f'{preview}...')

            print('-'*80)

        print('\n' + '='*80)

        # Show ranking comparison if reranked
        if reranked and original_results:
            print('\nRANKING COMPARISON:')
            print('-'*80)
            print('How reranking changed the order:\n')

            for idx, (result, rerank_score) in enumerate(results_to_display, 1):
                # Find original position
                original_pos = next(
                    (i for i, r in enumerate(original_results, 1) if r['id'] == result['id']),
                    None
                )

                if original_pos:
                    payload = result['payload']
                    filename = payload.get('file_path', '').split('/')[-1]

                    change = original_pos - idx
                    if change > 0:
                        change_str = f'↑ +{change}'
                    elif change < 0:
                        change_str = f'↓ {change}'
                    else:
                        change_str = '→ same'

                    print(f'{idx}. {filename[:50]:50} (was #{original_pos}) {change_str}')

            print('='*80)
