from types import SimpleNamespace

import pytest

from scripts.coreweave import rl_cloud_autopilot as autopilot


def test_entitlement_does_not_start_stale_recipe_while_launch_is_held(monkeypatch, tmp_path):
    monkeypatch.setattr(autopilot.rl_cloud, 'hub', lambda run: tmp_path)
    monkeypatch.setattr(autopilot, 'lineage_state', lambda *a: pytest.fail('held launch inspected stale smoke'))
    monkeypatch.setattr(autopilot, 'launch_on_entitled', lambda *a, **k: pytest.fail('held launch started GPU'))
    run = {'launch_hold': 'replace obsolete recipe with qualified current source'}
    autopilot.one_pass(run, SimpleNamespace(interval=3600, smoke='smoke', lineages=['production']),
                       {'entitled': ['account']})


def test_actual_launch_rereads_hold_before_spending(monkeypatch):
    monkeypatch.setattr(autopilot.rl_cloud, 'load_run', lambda path: {'launch_hold': 'current operator hold'})
    with pytest.raises(ValueError, match='current operator hold'):
        autopilot.rl_cloud.launch({'_path': 'run.json'}, 'production')
