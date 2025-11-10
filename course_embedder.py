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
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
from qdrant_client.models import Distance, VectorParams, PointStruct


class CourseEmbedder:

    def __init__(self, qdrant_host: str = 'localhost', qdrant_port: int = 6333, collection_name: str = 'course_materials', exclude_extensions: List[str] = None, exclude_filenames: List[str] = None):
        self.exclude_extensions = exclude_extensions or []
        self.exclude_filenames = exclude_filenames or []

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
        except:
            self.client.create_collection(
                collection_name = collection_name,
                vectors_config = VectorParams(size = 2560, distance = Distance.COSINE)
            )

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

                vector = self.embed_text(text_content)

                payload = {
                    'file_path': file_info['path'],
                    'file_type': file_info['type'],
                    'description': file_info.get('description', ''),

                    'group_id': group_id,
                    'group_type': group_data['group_type'],
                    'course': group_data['course'],

                    'sibling_files': sibling_paths,
                    'sibling_count': len(sibling_paths),

                    # preview for debugging purposes
                    'content_preview': text_content[:300],
                    'content_length': len(text_content)
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


if __name__ == '__main__':
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
