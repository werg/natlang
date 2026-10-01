"""Select the explicitly replayed current API corpus; historical artifacts remain inventoried."""
import hashlib
import json

def current_improvement_turns(repo):
    manifest_path = repo / 'data/teacher/self-improvement/current-manifest.json'
    if not manifest_path.exists():
        return []
    manifest = json.loads(manifest_path.read_text())
    selected = []
    for artifact in [*manifest['artifacts'], *manifest.get('historical_default_artifacts', [])]:
        path = repo / artifact['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
            raise ValueError(f'current improvement corpus changed: {path}')
        for evidence in artifact.get('producer_artifacts', []):
            source = repo / evidence['path']
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != evidence['sha256']:
                raise ValueError(f'historical improvement producer evidence changed: {source}')
        verification = artifact.get('verification_artifact')
        if verification:
            source = repo / verification
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != artifact.get('verification_sha256'):
                raise ValueError(f'historical improvement verification changed: {source}')
        review_manifest = artifact.get('review_manifest')
        if review_manifest:
            source = repo / review_manifest
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != artifact.get('review_manifest_sha256'):
                raise ValueError(f'historical improvement replacement review changed: {source}')
        selected.append(str(path.resolve()))
    return selected
