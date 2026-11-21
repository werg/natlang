from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'neuralese_data'))

import vibestudio_campaign


def test_failed_infrastructure_does_not_use_a_teacher_attempt(tmp_path):
    cases = {
        'reset': {'workspace': 'system', 'family': 'settings'},
        'android': {'workspace': 'system', 'family': 'device', 'requiresAndroidEmulator': True},
        'other': {'workspace': 'system', 'family': 'settings'},
        'recovered': {'workspace': 'system', 'family': 'settings'},
    }
    receipt = {
        'reset': {'status': 'test-failed', 'reason': '[eval.get] DO RPC EvalDO.getRun ECONNRESET'},
        'android': {'status': 'test-failed', 'reason': 'Connect exactly one ready disposable Android emulator'},
        'other': {'status': 'test-failed', 'reason': 'The requested update was not saved'},
        'recovered': {'status': 'export-rejected'},
    }
    rows = vibestudio_campaign.publish_status(
        tmp_path, cases, {'recovered': 'manifest.json'}, [receipt], None, 1,
        {'android': 'requires one booted disposable Android emulator'},
    )
    assert rows['reset']['attempts'] == 0
    assert rows['android']['attempts'] == 0
    assert rows['android']['status'] == 'prerequisite-unavailable'
    assert rows['other']['status'] == 'test-failed'
    assert rows['recovered']['status'] == 'verified'
    assert vibestudio_campaign.next_wave(cases, rows, 6) == ['reset']
    summary = vibestudio_campaign.read(tmp_path / 'status.json')
    assert summary['attempted_goals'] == summary['attempts'] == 4
    assert summary['infrastructure_failures'] == 2


def test_only_cases_with_unavailable_prerequisites_are_held(monkeypatch, tmp_path):
    cases = {
        'android': {'requiresAndroidEmulator': True},
        'deployment': {'requiresDeploymentPackage': True},
        'desktop': {'devices': ['desktop']},
    }
    monkeypatch.setattr(vibestudio_campaign.shutil, 'which', lambda name: name)

    def probes(command, **_kwargs):
        if command == ['adb', 'devices']:
            return subprocess.CompletedProcess(command, 0, stdout='emulator-5554\tdevice\nphysical\tdevice\n')
        if command[:2] == ['adb', '-s']:
            assert command[2] == 'emulator-5554'
            return subprocess.CompletedProcess(command, 0, stdout='1\n')
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr(vibestudio_campaign.subprocess, 'run', probes)
    assert set(vibestudio_campaign.prerequisite_holds(cases, tmp_path)) == {'deployment'}
