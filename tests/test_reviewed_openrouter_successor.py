import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'start_reviewed_openrouter_successor.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_creates_output_parents_before_claim(tmp_path):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from scripts.start_reviewed_openrouter_successor import preflight_successor_outputs

    plan = tmp_path / 'campaign' / 'plan.json'
    successor = {
        'launcher_log': str(plan.parent / 'worker' / 'launcher.log'),
        'journal': str(plan.parent / 'worker' / 'jobs' / 'journal.jsonl'),
        'status_file': str(plan.parent / 'worker-status.json'),
    }
    preflight_successor_outputs(successor)
    assert Path(successor['launcher_log']).parent.is_dir()
    assert Path(successor['journal']).parent.is_dir()
    assert Path(successor['status_file']).parent.is_dir()
    assert Path(successor['launcher_log']).is_file()


def test_output_parent_failure_does_not_claim_handoff(tmp_path):
    blocked = tmp_path / 'not-a-directory'
    blocked.write_text('file')
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    record = campaign / 'launch.json'
    plan_path = campaign / 'plan.json'
    plan = {
        'root_approved': True,
        'successor': {
            'launcher_log': str(blocked / 'launcher.log'),
            'journal': str(campaign / 'worker' / 'journal.jsonl'),
            'status_file': str(campaign / 'worker-status.json'),
        },
        'launch_record': str(record),
    }
    plan_path.write_text(json.dumps(plan))
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(plan_path), '--sha256', digest(plan_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert not record.exists()
    assert not record.with_suffix('.lock').exists()
