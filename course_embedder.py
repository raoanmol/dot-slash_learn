'''
Course Material Embedder

Embeds course materials (PDFs, code files, etc.) into Qdrant vector database
with group-based metadata for retrieval. Reads group definitions from JSON files
and creates embeddings using Qwen3-Embedding-4B model.

Usage:
    from course_embedder import CourseEmbedder

    embedder = CourseEmbedder(
        qdrant_host = 'localhost',
        qdrant_port = 6333,
        collection_name = 'course_materials',
        exclude_extensions=['.py', '.txt']
    )

    embedder.embed_course(
        mappings_json_path = 'data/computer_science/6_0001/groups.json',
        base_path = '.'
    )
'''

import os
import json
import torch
import PyPDF2
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForImageTextToText
from qdrant_client.models import Distance, VectorParams, PointStruct


class CourseEmbedder:

    def __init__(self, qdrant_host: str = 'localhost', qdrant_port: int = 6333, collection_name: str = 'course_materials', exclude_extensions: List[str] = None, exclude_filenames: List[str] = None, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize CourseEmbedder with semantic chunking support.

        Args:
            chunk_size: Target tokens per chunk (default: 500 ~= 700-800 words)
            chunk_overlap: Token overlap between chunks (default: 100 ~= 140 words)
        """
        self.exclude_extensions = exclude_extensions or []
        self.exclude_filenames = exclude_filenames or []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        if self.device.type == 'cuda':
            print(f'GPU: {torch.cuda.get_device_name(0)}')
            print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B')
        self.model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B')
        self.model.to(self.device)
        self.model.eval()

        self.client = QdrantClient(host = qdrant_host, port = qdrant_port)
        self.collection_name = collection_name

        try:
            self.client.get_collection(collection_name)
            print(f'Collection {collection_name} already exists')
        except Exception as e:
            print(f'Creating collection {collection_name}...')
            self.client.create_collection(
                collection_name = collection_name,
                vectors_config = VectorParams(size = 2560, distance = Distance.COSINE)
            )
            print(f'Collection {collection_name} created successfully')

        # LLM components for skill extraction/classification (lazy loaded)
        self.llm = None
        self.llm_processor = None

    def _load_llm(self):
        """Lazy load the LLM model for skill extraction and classification"""
        if self.llm is None:
            print('\nLoading Qwen3-VL-4B-Instruct for skill extraction...')
            self.llm_processor = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-4B-Instruct')
            self.llm = AutoModelForImageTextToText.from_pretrained(
                'Qwen/Qwen3-VL-4B-Instruct',
                dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map='auto'
            )
            self.llm.eval()
            print('LLM loaded successfully\n')

    def _extract_skills(self, combined_text: str, learning_objective: str, max_skills: int = 6) -> List[str]:
        """
        Extract high-level skills from course materials using LLM.

        Args:
            combined_text: Combined text from all documents
            learning_objective: The learning objective for these materials
            max_skills: Maximum number of skills to extract (default: 6)

        Returns:
            List of skill names
        """
        self._load_llm()

        # Truncate combined text if too long (keep first ~20k chars to fit in context)
        if len(combined_text) > 20000:
            combined_text = combined_text[:20000] + "\n\n[...truncated for length...]"

        prompt = f"""You are analyzing course materials to identify key skills and concepts being taught.

Learning Objective: {learning_objective}

Course Materials:
{combined_text}

Based on the learning objective and course materials above, identify {max_skills} high-level skills or concepts that are being taught. These should be:
- Specific and technical (not generic)
- Core concepts that appear in the materials
- Distinct from each other
- Relevant to the learning objective

Return ONLY a JSON array of skill names, nothing else. Example format:
["skill1", "skill2", "skill3", "skill4", "skill5"]

Skills:"""

        messages = [{"role": "user", "content": prompt}]

        text = self.llm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llm_processor(text=text, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,  # Low temperature for more consistent output
                top_p=0.9,
                do_sample=True
            )

        response = self.llm_processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Parse JSON response
        try:
            # Extract JSON array from response
            response = response.strip()
            # Find JSON array in response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                skills = json.loads(json_str)
                # Limit to max_skills
                return skills[:max_skills]
            else:
                print(f"Warning: Could not parse skills from LLM response: {response}")
                return []
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON from LLM response: {e}")
            print(f"Response was: {response}")
            return []

    def _classify_chunk(self, chunk: str, skills: List[str]) -> List[str]:
        """
        Classify a chunk into one or more skills.

        Args:
            chunk: The text chunk to classify
            skills: List of available skills

        Returns:
            List of skill names that apply to this chunk
        """
        if not skills:
            return []

        # Ensure LLM is loaded
        self._load_llm()

        prompt = f"""You are classifying a text chunk from course materials into relevant skills.

Available Skills:
{json.dumps(skills, indent=2)}

Text Chunk:
{chunk}

Which of the available skills does this text chunk teach or discuss? A chunk can cover multiple skills or none.

Return ONLY a JSON array of skill names that apply, nothing else. If no skills apply, return an empty array [].

Example format:
["skill1", "skill3"]

Classification:"""

        messages = [{"role": "user", "content": prompt}]

        text = self.llm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llm_processor(text=text, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.2,  # Very low temperature for consistent classification
                top_p=0.9,
                do_sample=True
            )

        response = self.llm_processor.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Parse JSON response
        try:
            response = response.strip()
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                classified_skills = json.loads(json_str)
                # Filter to only include valid skills
                return [s for s in classified_skills if s in skills]
            else:
                return []
        except json.JSONDecodeError:
            return []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + '\n'
                return text.strip()
        except Exception as e:
            print(f'Error extracting PDF {pdf_path}: {e}')
            return ''

    def extract_text_from_code(self, code_path: str) -> str:
        try:
            with open(code_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f'Error reading code file {code_path}: {e}')
            return ''

    def extract_text_from_file(self, file_path: str) -> str:
        if file_path.endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.endswith('.py'):
            return self.extract_text_from_code(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                print(f'Error reading file {file_path}: {e}')
                return ''

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks using sentence boundaries.

        Strategy:
        1. Split by sentences (using periods, exclamation marks, question marks)
        2. Group sentences into chunks targeting self.chunk_size tokens
        3. Overlap chunks by self.chunk_overlap tokens for context continuity

        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        # Split into sentences (simple but effective regex)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            # Tokenize to count tokens
            sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))

            # If adding this sentence exceeds chunk_size, finalize current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))

                # Start new chunk with overlap
                # Keep sentences from the end totaling ~chunk_overlap tokens
                overlap_chunk = []
                overlap_tokens = 0
                for sent in reversed(current_chunk):
                    sent_tokens = len(self.tokenizer.encode(sent, add_special_tokens=False))
                    if overlap_tokens + sent_tokens > self.chunk_overlap:
                        break
                    overlap_chunk.insert(0, sent)
                    overlap_tokens += sent_tokens

                current_chunk = overlap_chunk
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def embed_text(self, text: str) -> List[float]:
        if not text.strip():
            print('Warning: Empty text provided for embedding')
            return [0.0] * 2560

        inputs = self.tokenizer(text, return_tensors = 'pt', truncation = True, max_length = 8192, padding = True)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()

        return embedding.cpu().numpy().tolist()

    def load_mappings_from_json(self, json_path: str) -> Dict[str, Any]:
        with open(json_path, 'r') as f:
            return json.load(f)

    def embed_course(self, mappings_json_path: str, base_path: str = "."):
        groups = self.load_mappings_from_json(mappings_json_path)

        points = []
        point_id = self.get_next_point_id()

        for group_id, group_data in groups.items():
            if group_id.startswith('_TEMPLATE'):
                continue

            sibling_paths = [f['path'] for f in group_data['files']]

            for file_info in group_data['files']:
                file_path = os.path.join(base_path, file_info['path'])
                filename = os.path.basename(file_path)

                file_ext = os.path.splitext(file_path)[1]
                if file_ext in self.exclude_extensions:
                    print(f'- SKIPPED (extension {file_ext}): {filename}')
                    continue

                if filename in self.exclude_filenames:
                    print(f'- SKIPPED (filename): {filename}')
                    continue

                print(f'- Embedding {file_info['type']}: {filename}')

                text_content = self.extract_text_from_file(file_path)

                if not text_content.strip():
                    print(f'    WARNING: No content extracted!')
                    continue

                print(f'    Content length: {len(text_content)} characters')

                # Chunk the document
                chunks = self.chunk_text(text_content)
                print(f'    Split into {len(chunks)} chunks')

                # Create a point for each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    vector = self.embed_text(chunk)

                    payload = {
                        'file_path': file_info['path'],
                        'file_type': file_info['type'],
                        'description': file_info.get('description', ''),

                        'group_id': group_id,
                        'group_type': group_data['group_type'],
                        'course': group_data['course'],

                        'sibling_files': sibling_paths,
                        'sibling_count': len(sibling_paths),

                        # Chunk metadata
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'is_chunked': True,

                        # Store full chunk content for context
                        'full_content': chunk,
                        'content_preview': chunk[:300],
                        'content_length': len(chunk)
                    }

                    if group_data['group_type'] == 'lecture':
                        payload['lecture_number'] = group_data.get('lecture_number')
                        payload['title'] = group_data.get('title', '')
                    elif group_data['group_type'] == 'assignment':
                        payload['assignment_number'] = group_data.get('assignment_number')
                        payload['title'] = group_data.get('title', '')

                    point = PointStruct(id = point_id, vector = vector, payload = payload)

                    points.append(point)
                    point_id += 1

        if points:
            print(f'\n Uploading {len(points)} points to Qdrant...')
            batch_size = 10
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                print(f'  Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}')

            print(f'Successfully embedded {len(points)} documents!')
        else:
            print('No points to upload!')

    def get_next_point_id(self) -> int:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            points_count = collection_info.points_count
            return points_count
        except:
            return 0

    # ============================================================================
    # NEW API-based embedding methods
    # ============================================================================

    def embed_files(
        self,
        file_paths: List[str],
        learning_objective: str,
        course: str = None,
        group_type: str = None,
        group_id: str = None,
        **extra_metadata
    ) -> Dict[str, Any]:
        """
        Embed a list of files into the collection with skill-based classification.

        Pipeline:
        1. Extract text from all files
        2. Use LLM to extract 5-6 high-level skills from combined content
        3. Chunk all documents
        4. Parallel processing:
           - Classify each chunk into skills (multithreaded LLM calls)
           - Generate embeddings for each chunk (GPU batched)
        5. Upload to Qdrant with skill metadata

        Args:
            file_paths: List of absolute file paths to embed
            learning_objective: Description of what these files teach
            course: Optional course identifier (e.g., "6_0001")
            group_type: Optional group type (e.g., "lecture", "assignment")
            group_id: Optional group identifier
            **extra_metadata: Additional metadata to store with chunks

        Returns:
            Dict with success status, number of chunks created, skills extracted, and any errors
        """
        errors = []

        # ===================================================================
        # STEP 1: Extract text from all files
        # ===================================================================
        print('='*80)
        print('STEP 1: Extracting text from files')
        print('='*80)

        file_contents = []  # List of (file_path, text_content, file_type)

        for file_path in file_paths:
            filename = os.path.basename(file_path)

            # Check if file exists
            if not os.path.exists(file_path):
                errors.append(f"File not found: {file_path}")
                continue

            print(f'- Reading: {filename}')

            # Extract text from file
            text_content = self.extract_text_from_file(file_path)

            if not text_content.strip():
                errors.append(f"No content extracted from: {file_path}")
                print(f'    WARNING: No content extracted!')
                continue

            print(f'    Content length: {len(text_content)} characters')

            # Detect file type from extension
            file_ext = os.path.splitext(file_path)[1]
            file_type = 'code' if file_ext in ['.py', '.js', '.java', '.cpp', '.c'] else \
                       'pdf' if file_ext == '.pdf' else 'text'

            file_contents.append((file_path, text_content, file_type))

        if not file_contents:
            return {
                'success': False,
                'total_chunks': 0,
                'files_processed': 0,
                'errors': errors,
                'skills': []
            }

        # ===================================================================
        # STEP 2: Extract skills from combined content
        # ===================================================================
        print('\n' + '='*80)
        print('STEP 2: Extracting skills from course materials')
        print('='*80)

        # Combine all text content
        combined_text = "\n\n".join([content for _, content, _ in file_contents])
        print(f'Combined content length: {len(combined_text)} characters')

        # Extract skills using LLM
        skills = self._extract_skills(combined_text, learning_objective, max_skills=6)

        if skills:
            print(f'\nExtracted {len(skills)} skills:')
            for i, skill in enumerate(skills, 1):
                print(f'  {i}. {skill}')
        else:
            print('\nWarning: No skills extracted. Proceeding without skill classification.')

        # ===================================================================
        # STEP 3: Chunk all documents
        # ===================================================================
        print('\n' + '='*80)
        print('STEP 3: Chunking documents')
        print('='*80)

        all_chunks = []  # List of (file_path, chunk_text, chunk_idx, total_chunks, file_type)

        for file_path, text_content, file_type in file_contents:
            filename = os.path.basename(file_path)
            print(f'- Chunking: {filename}')

            chunks = self.chunk_text(text_content)
            print(f'    Split into {len(chunks)} chunks')

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append((file_path, chunk, chunk_idx, len(chunks), file_type))

        print(f'\nTotal chunks: {len(all_chunks)}')

        # ===================================================================
        # STEP 4: Parallel classification and embedding
        # ===================================================================
        print('\n' + '='*80)
        print('STEP 4: Classifying chunks and generating embeddings (parallel)')
        print('='*80)

        chunk_skills_map = {}  # Map chunk index to skills
        chunk_embeddings_map = {}  # Map chunk index to embedding vector

        # Function to classify a single chunk
        def classify_chunk_worker(idx_and_chunk):
            idx, (file_path, chunk, chunk_idx, total_chunks, file_type) = idx_and_chunk
            chunk_skills = self._classify_chunk(chunk, skills) if skills else []
            return idx, chunk_skills

        # Parallel classification using ThreadPoolExecutor
        if skills:
            print(f'\nClassifying {len(all_chunks)} chunks into skills...')
            with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust workers based on available resources
                future_to_idx = {
                    executor.submit(classify_chunk_worker, (idx, chunk_data)): idx
                    for idx, chunk_data in enumerate(all_chunks)
                }

                completed = 0
                for future in as_completed(future_to_idx):
                    idx, chunk_skills = future.result()
                    chunk_skills_map[idx] = chunk_skills
                    completed += 1
                    if completed % 10 == 0 or completed == len(all_chunks):
                        print(f'  Classified {completed}/{len(all_chunks)} chunks')
        else:
            print('\nSkipping classification (no skills extracted)')
            for idx in range(len(all_chunks)):
                chunk_skills_map[idx] = []

        # Generate embeddings (can be done in parallel with classification, but we'll do it after for simplicity)
        print(f'\nGenerating embeddings for {len(all_chunks)} chunks...')
        for idx, (file_path, chunk, chunk_idx, total_chunks, file_type) in enumerate(all_chunks):
            vector = self.embed_text(chunk)
            chunk_embeddings_map[idx] = vector
            if (idx + 1) % 10 == 0 or (idx + 1) == len(all_chunks):
                print(f'  Embedded {idx + 1}/{len(all_chunks)} chunks')

        # ===================================================================
        # STEP 5: Create points and upload to Qdrant
        # ===================================================================
        print('\n' + '='*80)
        print('STEP 5: Uploading to Qdrant')
        print('='*80)

        points = []
        point_id = self.get_next_point_id()

        for idx, (file_path, chunk, chunk_idx, total_chunks, file_type) in enumerate(all_chunks):
            vector = chunk_embeddings_map[idx]
            chunk_skills = chunk_skills_map[idx]

            # Build metadata payload
            payload = {
                'file_path': file_path,
                'file_type': file_type,
                'learning_objective': learning_objective,
                'skills': chunk_skills,  # Add skills to metadata

                # Chunk metadata
                'chunk_index': chunk_idx,
                'total_chunks': total_chunks,
                'is_chunked': True,

                # Store full chunk content for context
                'full_content': chunk,
                'content_preview': chunk[:300],
                'content_length': len(chunk)
            }

            # Add optional fields
            if course:
                payload['course'] = course
            if group_type:
                payload['group_type'] = group_type
            if group_id:
                payload['group_id'] = group_id

            # Add any extra metadata
            payload.update(extra_metadata)

            point = PointStruct(id=point_id, vector=vector, payload=payload)
            points.append(point)
            point_id += 1

        # Upload to Qdrant in batches
        if points:
            print(f'\nUploading {len(points)} points to Qdrant...')
            batch_size = 10
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                print(f'  Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}')

            print(f'\n✓ Successfully embedded {len(points)} chunks from {len(file_contents)} files!')

        # Offload LLM to free memory
        if self.llm is not None:
            print('\nOffloading LLM to CPU...')
            self.llm.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            'success': len(points) > 0,
            'total_chunks': len(points),
            'files_processed': len(file_contents),
            'errors': errors,
            'skills': skills
        }

    def delete_file(self, file_path: str) -> Dict[str, Any]:
        """
        Delete all chunks associated with a specific file from the collection.

        Args:
            file_path: Exact file path to match and delete

        Returns:
            Dict with success status and number of chunks deleted
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        try:
            # Search for all points with matching file_path
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=file_path)
                        )
                    ]
                ),
                limit=10000  # Adjust if you expect more chunks per file
            )

            points_to_delete = search_result[0]
            point_ids = [point.id for point in points_to_delete]

            if not point_ids:
                return {
                    'success': True,
                    'chunks_deleted': 0,
                    'message': f'No chunks found for file: {file_path}'
                }

            # Delete the points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )

            print(f'Deleted {len(point_ids)} chunks for file: {file_path}')

            return {
                'success': True,
                'chunks_deleted': len(point_ids),
                'message': f'Successfully deleted {len(point_ids)} chunks'
            }

        except Exception as e:
            print(f'Error deleting file {file_path}: {e}')
            return {
                'success': False,
                'chunks_deleted': 0,
                'error': str(e)
            }


# ============================================================================
# OLD JSON-based embedding logic (commented out for API-based system)
# ============================================================================
# if __name__ == '__main__':
#     embedder = CourseEmbedder(
#         qdrant_host = 'localhost',
#         qdrant_port = 6333,
#         collection_name = 'course_materials',
#         exclude_extensions=['.py', '.txt']
#     )
#
#     embedder.embed_course(
#         mappings_json_path = 'data/computer_science/6_0001/groups.json',
#         base_path = '.'
#     )
