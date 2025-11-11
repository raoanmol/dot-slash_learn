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


class LLMQuerySystem:

    def __init__(self, collection_name: str = 'cs_materials', qdrant_host: str = 'localhost', qdrant_port: int = 6333):
        """
        Initialize the LLM query system with RAG.

        Args:
            collection_name: Qdrant collection to search
            qdrant_host: Qdrant host address
            qdrant_port: Qdrant port
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

        # Load LLM
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

        Args:
            user_query: The user's question
            show_context: Whether to print the retrieved documents (for debugging)
            max_length: Maximum length of generated response

        Returns:
            LLM response string
        """
        # Stage 1 & 2: Perform RAG with two-stage reranking
        print('\n' + '='*80)
        print('RETRIEVING RELEVANT DOCUMENTS')
        print('='*80)

        documents = self.query_engine.search(
            query=user_query,
            top_k=3,
            use_reranker=True,
            rerank_candidates=10,
            verbose=True
        )

        if not documents:
            return "Sorry, I couldn't find any relevant documents to answer your question."

        # Format context
        context = self.format_context(documents)

        # Show retrieved documents for debugging
        if show_context:
            print('\n' + '='*80)
            print('RETRIEVED DOCUMENTS (Context for LLM)')
            print('='*80)
            print(context)

        # Build prompt
        prompt = f"""Based on the following course materials, please answer the question. Use the information from the documents to provide a comprehensive and accurate answer.

{context}

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

    args = parser.parse_args()

    # Initialize system
    system = LLMQuerySystem(collection_name=args.collection)

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
