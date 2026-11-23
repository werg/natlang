"""Source-bound correction of a review that confused baseline state with a task action.

Content is independently attested against exact retained documents; this module
never infers semantic acceptance from a phrase or changes the original review.
"""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes())


def bind(root, name, value):
    path = root / name
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return {'path': name, 'sha256': digest(path)}


def assess(packet_path, review_path, attestation_path, catalog_path, out):
    packet, review, attestation = map(read, (packet_path, review_path, attestation_path))
    observation = packet['execution']['diagnostics']['openGoalVerification']
    teacher = packet['execution']['diagnostics']['natlangTeacher']
    project = observation['preparedContentProject']
    original_input = project['artifact']['semanticReviewInput']
    documents = original_input['actual']['documents']
    document_digest = hashlib.sha256(json.dumps(documents, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    before, current = project['before'], project['actual']
    checks = observation['checks']
    if (packet['result'].get('passed') is not True or not checks or any(value is not True for value in checks.values()) or
            teacher.get('executionError') or teacher.get('captureErrors') or teacher.get('rootOutcome') != 'done' or
            project['expectedOutcome'] != 'draft' or project['artifact'].get('passed') is not True or
            current['committed'] != before['committed'] or current['clean'] is not False or
            current['workingHead'].get('kind') != 'application' or current['mainEventId'] != before['mainEventId'] or
            any(item['service'] == 'contentProjectCommit' and item['method'] == 'commit' and item.get('result') is not None for item in project['journal'])):
        raise ValueError('baseline correction requires a completed native private draft with no new commit')
    if (review.get('schema') != 'natlang.live-workflow-review/1' or review.get('input') != original_input or
            review.get('verdict', {}).get('accepted') is not False):
        raise ValueError('correction must preserve the rejected original review of these exact documents')
    if (attestation.get('schema') != 'natlang.source-bound-content-audit/1' or
            attestation.get('original_packet_sha256') != digest(packet_path) or
            attestation.get('original_review_sha256') != digest(review_path) or
            attestation.get('document_sha256') != document_digest or attestation.get('accepted') is not True or
            not attestation.get('rationale') or not attestation.get('auditor')):
        raise ValueError('semantic content audit must bind exact original evidence and its independent judgment')
    excerpt = attestation.get('original_content_positive_excerpt')
    if not isinstance(excerpt, str) or not excerpt or excerpt not in review['verdict'].get('verdict', ''):
        raise ValueError('original content-positive evidence must be quoted faithfully')
    facts = {fact['id']: fact['text'] for fact in original_input['expected']['facts']}
    coverage = attestation.get('fact_coverage') or []
    if len(coverage) != len(facts) or {item.get('fact_id') for item in coverage} != set(facts):
        raise ValueError('content audit must account for every supplied fact')
    by_path = {item['path']: item['text'] for item in documents}
    for item in coverage:
        if item.get('source_fact') != facts[item['fact_id']] or not item.get('rationale') or not item.get('quote') or item['quote'] not in by_path.get(item.get('path'), ''):
            raise ValueError('content audit support differs from retained source facts or document bytes')
    roles = attestation.get('role_coverage') or []
    if {item.get('role') for item in roles} != set(original_input['expected']['documentRoles']) or any(not item.get('rationale') or item.get('path') not in by_path for item in roles):
        raise ValueError('content audit must establish each requested document role')
    task = next(item for item in map(json.loads, Path(catalog_path).read_text().splitlines()) if item['scenario'] == packet['test']['name'])
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    copied_review = root / 'original-rejected-review.json'
    shutil.copyfile(review_path, copied_review)
    review_binding = {'path': copied_review.name, 'sha256': digest(copied_review), 'role': 'original-rejected-review'}
    content_binding = bind(root, 'independent-content-audit.json', attestation)
    lifecycle = {'baseline': before, 'current': current, 'delta': {'committedEventChanged': False, 'workingHeadChanged': True, 'protectedMainChanged': False},
                 'originalReviewAccepted': False, 'nativePassed': True, 'newTeacherInference': False, 'newReviewerInference': False}
    native_binding = bind(root, 'native-lifecycle-correction.json', lifecycle)
    checker = root / 'assessor.py'
    shutil.copyfile(__file__, checker)
    proof = {'schema': 'natlang.live-outcome-reassessment/1', 'task_id': task['taskId'], 'original_sha256': digest(packet_path),
             'goal': observation['goal'], 'entry': teacher.get('entry'), 'actualResult': teacher['report'], 'passed': True,
             'checks': checks, 'rationale': 'The independently audited retained documents satisfy the goal; native state proves the reviewer mistook an unchanged baseline event for a new commit. Original native success and negative review are both preserved.',
             'evidence': [review_binding, content_binding, native_binding],
             'checker_sources': [{'path': checker.name, 'sha256': digest(checker)}]}
    outcome = bind(root, 'completed-outcome.json', proof)
    assessment = {'schema': 'natlang.live-reassessment/1', 'task_id': task['taskId'], 'original_sha256': digest(packet_path),
                  'rationale': proof['rationale'], 'completed_outcome': outcome}
    bind(root, 'assessment.json', assessment)
    return assessment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('packet', 'review', 'attestation', 'catalog', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(assess(args.packet, args.review, args.attestation, args.catalog, args.out)))
