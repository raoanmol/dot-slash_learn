'''
LLM Query System with RAG

Main entry point for querying course materials with LLM assistance.
Performs two-stage reranking RAG and passes context to Qwen3-VL-4B-Instruct.

Usage:
    python llm.py "What is dynamic programming?"
    python llm.py "Explain Newton's laws" --collection physics_materials
'''

import argparse
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
from query import QueryEngine
from guardrails import SafetyGuardrails


class LLMQuerySystem:

    def __init__(self, collection_name: str = 'cs_materials', qdrant_host: str = 'localhost', qdrant_port: int = 6333, enable_guardrails: bool = True):
        """
        Initialize the LLM query system with RAG.

        Args:
            collection_name: Qdrant collection to search
            qdrant_host: Qdrant host address
            qdrant_port: Qdrant port
            enable_guardrails: Whether to enable safety guardrails
        """
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Initialize RAG query engine
        print('\n' + '='*80)
        print('INITIALIZING RAG SYSTEM')
        print('='*80)
        self.query_engine = QueryEngine(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            collection_name=collection_name
        )

        # Initialize guardrails (on CPU to save VRAM)
        self.enable_guardrails = enable_guardrails
        self.guardrails = None
        if self.enable_guardrails:
            print('\n' + '='*80)
            print('INITIALIZING GUARDRAILS')
            print('='*80)
            self.guardrails = SafetyGuardrails(device='cpu', verbose=True)

        # Don't load LLM yet - we'll lazy load it after RAG to save VRAM
        self.llm = None
        self.processor = None
        print('\nLLM will be loaded after RAG completes (lazy loading to save VRAM)\n')

    def _load_llm(self):
        """Lazy load the LLM model when needed"""
        if self.llm is None:
            print('\n' + '='*80)
            print('LOADING LLM: Qwen3-VL-4B-Instruct')
            print('='*80)
            self.processor = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-4B-Instruct')
            self.llm = AutoModelForImageTextToText.from_pretrained(
                'Qwen/Qwen3-VL-4B-Instruct',
                dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map='auto'
            )
            self.llm.eval()
            print(f'LLM loaded successfully\n')

    def _rewrite_query(self, original_query: str) -> str:
        """
        Rewrite the user query to be more detailed and contextually rich for RAG.

        This helps improve similarity search by expanding simple questions into
        more comprehensive queries that match better with course material embeddings.

        Args:
            original_query: The user's original question

        Returns:
            Rewritten query optimized for RAG retrieval
        """
        # Load LLM if not already loaded (should be on GPU)
        self._load_llm()

        # Build prompt for query rewriting with better instructions
        rewrite_prompt = f"""You are improving search accuracy for a Retrieval-Augmented Generation (RAG) system that searches computer science course materials.

The RAG system uses vector similarity search - it embeds queries and documents, then finds the most similar documents. Simple queries often don't match well with technical course content.

Your task: Rewrite the user's question to include MORE technical terminology, synonyms, related concepts, and CS-specific vocabulary. This will improve the embedding similarity with course materials.

Original question: "{original_query}"

Guidelines:
- Add technical terms and synonyms (e.g., "OOP" → "object-oriented programming, OOP, encapsulation, inheritance, polymorphism")
- Include related concepts that appear in course materials
- Expand abbreviations and add full terms
- Keep the same meaning but make it more detailed and technical
- Make it sound like how a CS textbook or lecture would phrase it

Examples:
Input: "What is DP?"
Output: "dynamic programming algorithms, DP optimization technique, memoization and tabulation, optimal substructure property, overlapping subproblems, recursive solutions with caching"

Input: "How do loops work?"
Output: "iteration and loops in programming, for loops and while loops, loop control flow, iterative structures, loop counters and conditions, nested loops and loop execution"

Now rewrite this query for better RAG search results:"""

        # Format as conversation
        messages = [
            {
                "role": "user",
                "content": rewrite_prompt
            }
        ]

        # Apply chat template and tokenize
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate rewritten query
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=100,  # Keep it concise
                temperature=0.5,      # Moderate temperature
                top_p=0.9,
                do_sample=True
            )

        # Decode response (skip the input prompt tokens)
        rewritten = self.processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Clean up the response
        rewritten = rewritten.strip()

        # Remove any common prefixes the model might add
        prefixes_to_remove = ['Output:', 'Expanded query:', 'Answer:', 'A:', 'Q:']
        for prefix in prefixes_to_remove:
            if rewritten.startswith(prefix):
                rewritten = rewritten[len(prefix):].strip()

        # Take only the first paragraph/line if multiple were generated
        if '\n' in rewritten:
            rewritten = rewritten.split('\n')[0].strip()

        # If rewriting failed or produced garbage, use original query
        if len(rewritten) < 3 or len(rewritten) > 500:
            return original_query

        return rewritten

    def format_context(self, documents: List[Tuple[Dict[str, Any], float]]) -> str:
        """
        Format retrieved documents as context for the LLM.

        Args:
            documents: List of (result_dict, score) tuples

        Returns:
            Formatted context string
        """
        context_parts = []

        for idx, (result, score) in enumerate(documents, 1):
            payload = result['payload']

            # Extract key information
            file_path = payload.get('file_path', 'N/A')
            file_type = payload.get('file_type', 'N/A')
            course = payload.get('course', 'N/A')

            # Get content (prefer full_content if available, otherwise use content_preview)
            content = payload.get('full_content', payload.get('content_preview', ''))

            # Format document
            doc_str = f"Document {idx} (Relevance: {score:.4f})\n"
            doc_str += f"Source: {file_path}\n"
            doc_str += f"Course: {course}\n"
            doc_str += f"Type: {file_type}\n"
            doc_str += f"Content:\n{content}\n"

            context_parts.append(doc_str)

        return '\n' + ('='*80) + '\n\n'.join(context_parts) + '\n' + ('='*80)

    def query(self, user_query: str, show_context: bool = True, max_length: int = 2048):
        """
        Query the LLM with RAG context.

        VRAM Management Strategy:
        1. Guardrails (CPU) + LLM (GPU) for input validation and query rewriting
        2. Offload guardrails + LLM, load embedder + reranker for RAG
        3. Offload RAG models, reload guardrails + LLM for answer generation

        Args:
            user_query: The user's question
            show_context: Whether to print the retrieved documents (for debugging)
            max_length: Maximum length of generated response

        Returns:
            LLM response string
        """
        # PHASE 1: INPUT VALIDATION & QUERY REWRITING (Guardrails + LLM on GPU)
        # ============================================================================

        # Move guardrails to GPU for faster validation
        if self.enable_guardrails:
            print('\n' + '='*80)
            print('LOADING GUARDRAILS to GPU...')
            print('='*80)
            self.guardrails.move_to_gpu()
            print('Guardrails ready on GPU\n')

        # GUARDRAIL CHECKPOINT 1: Validate input query
        if self.enable_guardrails:
            print('\n' + '='*80)
            print('GUARDRAIL CHECK: Validating Input Query')
            print('='*80)

            is_safe, reason = self.guardrails.validate_input(user_query)

            if not is_safe:
                print(f'\n✗ QUERY BLOCKED: {reason}')
                print('='*80)
                return f"I cannot process this query. Reason: {reason}"

            print('='*80)

        # QUERY REWRITING: Expand query for better RAG retrieval
        print('\n' + '='*80)
        print('QUERY REWRITING: Expanding query for RAG')
        print('='*80)
        print(f'Original query: "{user_query}"')

        rewritten_query = self._rewrite_query(user_query)
        print(f'Rewritten query: "{rewritten_query}"')
        print('='*80)

        # PHASE 2: RAG RETRIEVAL (Embedder + Reranker on GPU)
        # ============================================================================

        # Offload guardrails and LLM to CPU to free VRAM for RAG models
        print('\n' + '='*80)
        print('FREEING VRAM: Moving Guardrails + LLM to CPU for RAG...')
        print('='*80)

        if self.enable_guardrails:
            self.guardrails.offload_to_cpu()

        if self.llm is not None:
            self.llm.to('cpu')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('VRAM freed\n')

        # Stage 1 & 2: Perform RAG with two-stage reranking using rewritten query
        print('\n' + '='*80)
        print('RETRIEVING RELEVANT DOCUMENTS')
        print('='*80)

        documents = self.query_engine.search(
            query=rewritten_query,  # Use rewritten query for embedding
            top_k=3,               # Final number of documents to pass to LLM
            use_reranker=True,
            rerank_candidates=50,  # Retrieve 50 candidates from vector search
            stage1_top_k=7,        # Keep top 7 after first rerank, then expand with neighbors
            min_score=7.0,         # Minimum reranker score threshold
            verbose=True
        )

        # Allow LLM to answer without RAG if no documents pass the threshold
        use_rag = len(documents) > 0

        # PHASE 3: ANSWER GENERATION (LLM + Guardrails on GPU)
        # ============================================================================

        # Offload RAG models to CPU to free VRAM for LLM
        print('\n' + '='*80)
        print('FREEING VRAM: Moving RAG models to CPU...')
        print('='*80)
        self.query_engine.model.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('VRAM freed\n')

        # Format context if we have documents
        if use_rag:
            context = self.format_context(documents)

            # Show retrieved documents for debugging
            if show_context:
                print('\n' + '='*80)
                print('RETRIEVED DOCUMENTS (Context for LLM)')
                print('='*80)
                print(context)
        else:
            print('\n' + '='*80)
            print('⚠ NO DOCUMENTS PASSED THRESHOLD - LLM will answer without RAG')
            print('='*80)

        # Reload LLM and Guardrails to GPU for answer generation
        print('\n' + '='*80)
        print('RELOADING LLM + GUARDRAILS to GPU for answer generation...')
        print('='*80)

        # Move LLM back to GPU
        if self.llm is not None:
            self.llm.to(self.device)
        else:
            self._load_llm()

        # Move guardrails back to GPU
        if self.enable_guardrails:
            self.guardrails.move_to_gpu()

        print('LLM + Guardrails ready on GPU\n')

        # Build prompt using ORIGINAL user query (not rewritten)
        if use_rag:
            prompt = f"""Based on the following course materials, please answer the question. Use the information from the documents to provide a comprehensive and accurate answer.

{context}

Question: {user_query}

Answer:"""
        else:
            prompt = f"""Please answer the following question to the best of your knowledge. Note that no relevant course materials were found with high enough confidence, so provide a general answer based on your training.

Question: {user_query}

Answer:"""

        # Generate response
        print('\n' + '='*80)
        print('GENERATING RESPONSE')
        print('='*80 + '\n')

        # Format as conversation for the vision-language model
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Apply chat template and tokenize
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Extract just the answer part (after "Answer:")
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()

        # GUARDRAIL CHECKPOINT 2: Validate output response
        if self.enable_guardrails:
            print('\n' + '='*80)
            print('GUARDRAIL CHECK: Validating Output Response')
            print('='*80)

            # Extract content from documents for grounding check (if we have any)
            context_texts = []
            if use_rag:
                for result, score in documents:
                    payload = result['payload']
                    content = payload.get('full_content', payload.get('content_preview', ''))
                    context_texts.append(content)

            is_safe, reason, sanitized_response = self.guardrails.validate_output(
                query=user_query,
                response=response,
                context=context_texts if use_rag else None,
                check_hallucination=use_rag,  # Only check hallucination if we have context
                check_pii=True
            )

            if not is_safe:
                print(f'\n✗ RESPONSE BLOCKED: {reason}')
                print('='*80)
                return f"I generated a response, but it failed safety validation. Reason: {reason}"

            # Use sanitized response if PII was redacted
            if sanitized_response:
                response = sanitized_response

            print('='*80)

        return response


def main():
    parser = argparse.ArgumentParser(description='Query course materials with LLM assistance')
    parser.add_argument(
        'query',
        type=str,
        help='The question to ask'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default='cs_materials',
        help='Qdrant collection to search (default: cs_materials)'
    )
    parser.add_argument(
        '--no-context',
        action='store_true',
        help='Hide the retrieved documents (only show final answer)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help='Maximum length of generated response (default: 2048)'
    )
    parser.add_argument(
        '--no-guardrails',
        action='store_true',
        help='Disable safety guardrails (not recommended)'
    )

    args = parser.parse_args()

    # Initialize system
    system = LLMQuerySystem(
        collection_name=args.collection,
        enable_guardrails=not args.no_guardrails
    )

    # Query
    response = system.query(
        args.query,
        show_context=not args.no_context,
        max_length=args.max_length
    )

    # Print final answer
    print('='*80)
    print('FINAL ANSWER')
    print('='*80)
    print(response)
    print('\n' + '='*80)


if __name__ == '__main__':
    main()
