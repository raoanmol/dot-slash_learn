'''
Embed all subject materials into separate Qdrant collections

Each subject can have MULTIPLE courses (each with its own coursework_mappings.json file).
All courses for a subject are embedded into the SAME collection.

Example:
    Computer Science collection (cs_materials) contains:
        - 6.0001 (Intro to CS)
        - 6.006 (Algorithms)
        - 6.046J (Design and Analysis)
        - 6.S096 (Effective Programming)

Usage:
    python embed_all_subjects.py                        # Embed all subjects
    python embed_all_subjects.py --subjects cs physics  # Only specific subjects
'''

import argparse
from pathlib import Path
from course_embedder import CourseEmbedder

SUBJECT_CONFIG = {
    'cs': {
        'collection': 'cs_materials',
        'groups_paths': [
            'data/computer_science/6_0001/coursework_mappings.json',
            'data/computer_science/6_006/coursework_mappings.json'
        ],
        'display_name': 'Computer Science'
    },
    'physics': {
        'collection': 'physics_materials',
        'groups_paths': [
            'data/physics/8_01/groups.json',
            'data/physics/8_02/groups.json',
        ],
        'display_name': 'Physics'
    },
    'chemistry': {
        'collection': 'chemistry_materials',
        'groups_paths': [
            'data/chemistry/5_111/groups.json',
            'data/chemistry/5_112/groups.json',
        ],
        'display_name': 'Chemistry'
    },
    'math': {
        'collection': 'mathematics_materials',
        'groups_paths': [
            'data/mathematics/18_01/groups.json',
            'data/mathematics/18_02/groups.json',
        ],
        'display_name': 'Mathematics'
    },
    'biology': {
        'collection': 'biology_materials',
        'groups_paths': [
            'data/biology/7_012/groups.json',
        ],
        'display_name': 'Biology'
    },
    'ee': {
        'collection': 'ee_materials',
        'groups_paths': [
            'data/electrical_engineering/6_002/groups.json',
        ],
        'display_name': 'Electrical Engineering'
    },
    'me': {
        'collection': 'me_materials',
        'groups_paths': [
            'data/mechanical_engineering/2_001/groups.json',
        ],
        'display_name': 'Mechanical Engineering'
    }
}


def embed_subject(subject_key, config, qdrant_host = 'localhost', qdrant_port = 6333, exclude_extensions = None):
    print(f"\n{'='*70}")
    print(f"Embedding {config['display_name']} ({subject_key})")
    print(f"{'='*70}\n")

    groups_paths = config.get('groups_paths') or [config.get('groups_path')]

    existing_paths = []
    for path_str in groups_paths:
        path = Path(path_str)
        if path.exists():
            existing_paths.append(path)
        else:
            print(f'WARNING: {path} not found - skipping this file')

    if not existing_paths:
        print(f'ERROR: No valid groups.json files found for {subject_key}')
        return False

    print(f'Found {len(existing_paths)} course(s) to embed:')
    for path in existing_paths:
        print(f'  - {path}')
    print()

    try:
        embedder = CourseEmbedder(
            qdrant_host = qdrant_host,
            qdrant_port = qdrant_port,
            collection_name = config['collection'],
            exclude_extensions = exclude_extensions or []
            # Old behavior: Exclude code and text files
            # exclude_extensions = exclude_extensions or ['.py', '.txt', '.js', '.java', '.cpp', '.c']
        )

        total_embedded = 0
        for groups_path in existing_paths:
            course_name = groups_path.parent.name
            print(f'  Embedding course: {course_name} ({groups_path.name})')

            embedder.embed_course(
                mappings_json_path=str(groups_path),
                base_path='.'
            )
            total_embedded += 1
            print(f"   Completed {course_name}\n")

        print(f'Successfully embedded {total_embedded} course(s) for {config['display_name']}')
        return True

    except Exception as e:
        print(f'\nError embedding {config['display_name']}: {e}')
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description = 'Embed all subject materials into Qdrant')
    parser.add_argument(
        '--subjects',
        nargs = '+',
        choices = list(SUBJECT_CONFIG.keys()) + ['all'],
        default = ['all'],
        help = 'Subjects to embed (default: all)'
    )
    parser.add_argument('--host', type = str, default = 'localhost', help = 'Qdrant host')
    parser.add_argument('--port', type = int, default = 6333, help = 'Qdrant port')
    parser.add_argument(
        '--exclude-extensions',
        nargs = '+',
        default = [],  # Embed everything by default
        # Old default: ['.py', '.txt', '.js', '.java', '.cpp', '.c']
        help = 'File extensions to exclude (default: none, embed everything)'
    )

    args = parser.parse_args()

    if 'all' in args.subjects:
        subjects_to_embed = list(SUBJECT_CONFIG.keys())
    else:
        subjects_to_embed = args.subjects

    print('\n' + '='*70)
    print('COURSE MATERIAL EMBEDDING')
    print('='*70)
    print(f'Qdrant: {args.host}:{args.port}')
    print(f'Subjects: {', '.join([SUBJECT_CONFIG[s]['display_name'] for s in subjects_to_embed])}')
    print(f'Excluding: {', '.join(args.exclude_extensions)}')
    print('='*70)

    results = {}
    for subject_key in subjects_to_embed:
        config = SUBJECT_CONFIG[subject_key]
        success = embed_subject(
            subject_key,
            config,
            qdrant_host=args.host,
            qdrant_port=args.port,
            exclude_extensions=args.exclude_extensions
        )
        results[subject_key] = success

    print('\n' + '='*70)
    print('EMBEDDING SUMMARY')
    print('='*70)

    successful = [s for s, success in results.items() if success]
    failed = [s for s, success in results.items() if not success]

    if successful:
        print(f'\n Successfully embedded ({len(successful)}):')
        for subject in successful:
            print(f'  - {SUBJECT_CONFIG[subject]['display_name']}')

    if failed:
        print(f'\nFailed to embed ({len(failed)}):')
        for subject in failed:
            print(f'  - {SUBJECT_CONFIG[subject]['display_name']}')

    print('\n' + '='*70)

    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(host=args.host, port=args.port)
        collections = client.get_collections()

        print('\nQdrant Collections:')
        for collection in collections.collections:
            info = client.get_collection(collection.name)
            print(f'  - {collection.name}: {info.points_count} documents')

    except Exception as e:
        print(f'\nCould not verify collections: {e}')

    print('\n' + '='*70)
    print('DONE!')
    print('='*70 + '\n')


if __name__ == '__main__':
    main()
