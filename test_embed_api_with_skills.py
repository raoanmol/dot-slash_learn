'''
Test script for the updated embed API with skill extraction and classification.

This script tests the new pipeline:
1. Extract skills from course materials
2. Classify chunks into skills
3. Store embeddings with skill metadata

Usage:
    python test_embed_api_with_skills.py
'''

import requests
import json

# API endpoint
API_URL = "http://localhost:8000"

def test_embed_with_skills():
    """Test the embed endpoint with skill extraction"""

    # Sample request - you should replace these with actual file paths
    request_data = {
        "file_paths": [
            "/home/exouser/pylti1.3-fastapi/downloaded_materials/course-5/Lecture-1/1762932458_867__904de062253d2fad3064ab5ca917883d_nykOeWgQcHM.pdf",  # Replace with actual path
            "/home/exouser/pylti1.3-fastapi/downloaded_materials/course-5/Lecture-1/1762933012_588__e921a690079369751bcce3e34da6c6ee_MIT6_0001F16_Lec1.pdf"
        ],
        "learning_objective": "Introduction to Programming in Python",
        "collection_name": "test_skills_collection",
        "course": "CS101",
        "group_type": "lecture",
        "group_id": "lecture_01"
    }

    print("=" * 80)
    print("Testing Embed Endpoint with Skills")
    print("=" * 80)
    print(f"\nRequest:")
    print(json.dumps(request_data, indent=2))

    print("\nSending request to API...")

    try:
        response = requests.post(
            f"{API_URL}/api/v1/embed",
            json=request_data,
            timeout=300  # 5 minute timeout for LLM processing
        )

        print(f"\nResponse Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\n" + "=" * 80)
            print("SUCCESS!")
            print("=" * 80)
            print(f"\nTotal chunks created: {result['total_chunks']}")
            print(f"Files processed: {result['files_processed']}")
            print(f"\nExtracted Skills ({len(result['skills'])}):")
            for i, skill in enumerate(result['skills'], 1):
                print(f"  {i}. {skill}")

            if result['errors']:
                print(f"\nErrors:")
                for error in result['errors']:
                    print(f"  - {error}")

            print(f"\nMessage: {result['message']}")
        else:
            print("\nERROR:")
            print(response.text)

    except requests.exceptions.Timeout:
        print("\nERROR: Request timed out. The LLM processing may take longer than expected.")
    except Exception as e:
        print(f"\nERROR: {e}")

def test_health():
    """Test the health endpoint"""
    print("\n" + "=" * 80)
    print("Testing Health Endpoint")
    print("=" * 80)

    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"ERROR: {e}")

def test_query_with_skills():
    """Test querying with skill-based filtering (if implemented)"""
    print("\n" + "=" * 80)
    print("Testing Query Endpoint")
    print("=" * 80)

    request_data = {
        "query": "What is inheritance?",
        "collection_name": "test_skills_collection",
        "show_context": True
    }

    try:
        response = requests.post(
            f"{API_URL}/api/v1/query",
            json=request_data,
            timeout=60
        )

        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"\nAnswer: {result['answer']}")

            if result.get('context'):
                print(f"\nContext documents ({len(result['context'])}):")
                for i, doc in enumerate(result['context'], 1):
                    print(f"\n  Document {i}:")
                    print(f"    File: {doc['file_path']}")
                    print(f"    Skills: {doc.get('skills', 'N/A')}")
                    print(f"    Relevance: {doc['relevance_score']:.4f}")
        else:
            print(f"ERROR: {response.text}")

    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    # Test health first
    test_health()

    # Test embedding with skills
    print("\n" + "=" * 80)
    print("IMPORTANT: Update the file_paths in this script with actual paths before running!")
    print("=" * 80)

    # Uncomment to test embedding (after updating file paths)
    test_embed_with_skills()

    # Uncomment to test querying (after embedding some documents)
    # test_query_with_skills()

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
