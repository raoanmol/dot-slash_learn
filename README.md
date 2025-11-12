# dot-slash_learn

RAG-LLM Query API for course materials.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set Qdrant connection (optional, defaults to localhost:6333)
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
```

## Run

```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

## Usage

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is dynamic programming?",
    "collection_name": "cs_materials"
  }'
```

**Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={"query": "What is recursion?", "collection_name": "cs_materials"}
)

print(response.json()["answer"])
```

## API

**POST /api/v1/query**

```json
{
  "query": "string (required)",
  "collection_name": "string (default: cs_materials)",
  "show_context": "boolean (default: false)",
  "max_length": "integer (default: 2048)",
  "enable_guardrails": "boolean (default: true)"
}
```

**GET /health**

Check if API and Qdrant are connected.
