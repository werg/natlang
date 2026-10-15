"""S7 `neuralese_rl` stage: RFT data selection, recipe wiring onto the shared direct trainer, argument forwarding."""
import json

import pytest

from natlang_neuralese.rl import stage as rl_stage
from natlang_neuralese.rl.stage import held_task, select_rft_documents
from natlang_neuralese.train import recipe


def _rollout(task, quality, turns=1):
    turn = {"messages": [{"role": "user", "content": task}], "tools": None,
            "target": {"role": "assistant", "content": "ok"}}
    return {"task_id": task, "reward": None if quality is None else {"quality": quality}, "turns": [turn] * turns}


def test_rft_selects_successes_of_sometimes_solved_tasks_only():
    rollouts = ([_rollout("mixed", 1.0, 2), _rollout("mixed", 0.0), _rollout("mixed", None)]
                + [_rollout("always", 1.0)] * 4 + [_rollout("never", 0.0)] * 4)
    documents, receipt = select_rft_documents(rollouts, max_success_rate=0.75, held_fraction=0.0)
    assert {d["task_id"] for d in documents} == {"mixed"} and len(documents) == 2  # two turns of one success
    assert receipt["skipped"] == {"never_solved": 1, "always_solved_above_rate": 1}
    assert receipt["tasks_selected"][0]["success_rate"] == 0.5  # the unscored rollout does not count


def test_held_split_is_by_task_and_deterministic():
    tasks = [f"task-{i}" for i in range(400)]
    held = [t for t in tasks if held_task(t, 0.125)]
    assert held == [t for t in tasks if held_task(t, 0.125)] and 20 < len(held) < 80
    assert not any(held_task(t, 0.0) for t in tasks)


def test_rl_handler_extends_the_direct_trainer():
    rl, direct = recipe.HANDLERS["neuralese_rl"], recipe.HANDLERS["neuralese_direct"]
    assert direct["parameters"] <= rl["parameters"] and recipe.RL_PARAMETERS <= rl["parameters"]
    assert recipe.missing_stage_inputs("neuralese_rl", {"rollouts"}) == {
        "continue_from", "continue_identity", "continue_monitor_reference"}
    lineage = {"stages": [{"id": "direct_sft", "kind": "neuralese_direct", "requires": []},
                          {"id": "rl_rft", "kind": "neuralese_rl", "requires": ["direct_sft"],
                           "inputs": {"rollouts": "s7_rollouts", "heads": "base_heads"}}]}
    assert recipe.direct_phase_predecessor(lineage, lineage["stages"][1]) == "direct_sft"
    recipe.check_direct_phase_continuation(lineage, lineage["stages"][1])  # continuation roles bound implicitly
    parameters = recipe.effective_stage_parameters(
        {"kind": "neuralese_rl", "parameters": {"rft_max_success_rate": 0.5}}, [],
        {"stage_parameter_defaults": {"neuralese_direct": {"latent_lr": 3e-6, "epochs": 1.0},
                                      "neuralese_rl": {"epochs": 2.0}}})
    assert parameters == {"latent_lr": 3e-6, "epochs": 2.0, "rft_max_success_rate": 0.5}
    assert "rollouts" in " ".join(recipe.stage_input_args({"rollouts": {"path": "/r.jsonl"}}, "neuralese_rl"))


def test_stage_forwards_to_direct_trainer_with_fresh_reference(tmp_path, monkeypatch):
    rollouts = tmp_path / "r.jsonl"
    rows = [_rollout(f"t{i}", q) for i in range(40) for q in (1.0, 0.0)]
    rollouts.write_text("".join(json.dumps(r) + "\n" for r in rows))
    calls = {}
    def render(documents, tokenizer, mode):
        calls['rendering'] = mode
        return [{"text": "x" + d["task_id"], "split": d["split"],
                 "source_groups": [d["task_id"]]} for d in documents]
    monkeypatch.setattr(rl_stage, "render_rows", render)
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda path: object()))
    from natlang_neuralese.train import direct_sft
    monkeypatch.setattr(direct_sft, "main", lambda argv: calls.setdefault("argv", argv))
    rl_stage.main(["--rollouts", str(rollouts), "--rl-tokenizer", str(tmp_path), "--out", str(tmp_path / "out"),
                   "--heads", "/h.pt", "--continue-from", "/c.pt", "--epochs", "1"])
    argv = calls["argv"]
    assert argv[:6] == ["--heads", "/h.pt", "--continue-from", "/c.pt", "--epochs", "1"]
    assert "--fresh-monitor-reference" in argv and argv[argv.index("--text-data") + 1].endswith("rl-text-data.jsonl")
    receipt = json.loads((tmp_path / "out" / "rl-data-receipt.json").read_text())
    assert receipt["documents"] == 40 and len(receipt["tasks_selected"]) == 40
    assert calls['rendering'] == 'full'  # preserve the managed actor's system text and tool schemas
    assert receipt['training_prompt_source'] == 'full'


def test_stage_refuses_without_train_and_held_documents(tmp_path):
    rollouts = tmp_path / "r.jsonl"
    rollouts.write_text(json.dumps(_rollout("never", 0.0)) + "\n")
    with pytest.raises(SystemExit):
        rl_stage.main(["--rollouts", str(rollouts), "--rl-tokenizer", str(tmp_path), "--out", str(tmp_path / "o")])


def test_rft_objective_weights_successes_of_selected_tasks(monkeypatch):
    from natlang_neuralese.rl import grpo

    class FakeServer:
        def __init__(self):
            self.terms = None

        def sequence_logps(self, rollouts, adapters):
            return [0.0] * len(rollouts)

        def post(self, path, body):
            if path.endswith("/grad"):
                self.terms = body["terms"]
                return {"loss": 0.0, "gradients": {"a0": "g"}}
            return {"params": ["a1"], "state": {}}

    rollouts = [dict(_rollout("mixed", 1.0), group=0), dict(_rollout("mixed", 0.0), group=0),
                dict(_rollout("always", 1.0), group=1), dict(_rollout("always", 1.0), group=1)]
    server = FakeServer()
    result = grpo.run_round(server, rollouts, "a0", {}, lr=1e-3, beta=0.0, clip=0.2, steps=1, objective="rft")
    assert result["report"]["rft_tasks"] == 1 and result["adapter"] == "a1"
    assert len(server.terms) == 1 and server.terms[0]["weight"] == 1.0  # only the mixed task's success


def test_grpo_stage_binds_policy_replays_and_shared_direct_trainer(tmp_path, monkeypatch):
    import hashlib
    import transformers
    from natlang_neuralese.train import direct_sft
    checkpoint = tmp_path / 'parent.pt'
    checkpoint.write_bytes(b'full checkpoint')
    policy = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows = []
    for i in range(50):
        for q in (1., 0.):
            r = _rollout(f't{i}', q)
            r.update(split='train', group=i, sample=int(q),
                     policy={'checkpoint_sha256': policy, 'temperature': 1.})
            r['turns'][0]['replay'] = {'raw_calls_retained': True}
            r['turns'][0]['sampling'] = {'schema': 'natlang.rl-sampling/1', 'guided': False,
                'temperature': 1., 'prompt': [['tokens', [1]]], 'items': [['tok', 2]],
                'samples': [{'item': 0, 'token': 2, 'logp': -1.}]}
            rows.append(r)
    source = tmp_path / 'rollouts.jsonl'
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda path: object())
    from natlang_neuralese.data import text_corpus
    monkeypatch.setattr(text_corpus, 'tokenizer_fingerprint', lambda _: 'fp')
    monkeypatch.setattr(rl_stage, 'render_rows', lambda docs, *_: [
        {'text': d['task_id'] + d['target']['content'], 'split': d['split'],
         'source_groups': [d['task_id']]} for d in docs])
    calls = []
    monkeypatch.setattr(direct_sft, 'main', lambda argv: calls.append(argv))
    out = tmp_path / 'out'
    rl_stage.main(['--rollouts', str(source), '--rl-tokenizer', str(tmp_path), '--out', str(out),
                   '--rl-mode', 'grpo', '--rl-policy-id', policy,
                   '--continue-from', str(checkpoint)])
    assert '--rl-objective' in calls[0]
    spec = json.loads((out / 'rl-objective.json').read_text())
    assert spec['policy_id'] == policy
    # Identical sampled actions on successful/failed trajectories cancel rather
    # than silently teaching one randomly retained advantage.
    assert spec['documents'] and sum(d['advantage'] for d in spec['documents']) == pytest.approx(0.)
    assert all(d['multiplicity'] == 1 for d in spec['documents'])
    assert len({d['document_sha256'] for d in spec['documents']}) == len(spec['documents'])


def test_registered_production_held_rows_are_evaluation_only(tmp_path, monkeypatch):
    import transformers
    from natlang_neuralese.train import direct_sft
    source = tmp_path / 'rollouts.jsonl'
    source.write_text(''.join(json.dumps(_rollout('task', q)) + '\n' for q in [1., 0.]))
    held = tmp_path / 'held.jsonl'
    held.write_text(json.dumps({'split': 'train', 'text': 'must not copy'}) + '\n' +
                    json.dumps({'split': 'test', 'text': 'protected', 'source_groups': ['held']}) + '\n')
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda path: object())
    monkeypatch.setattr(rl_stage, 'render_rows', lambda documents, *_: [
        {'split': d['split'], 'text': d['task_id'], 'source_groups': [d['task_id']]} for d in documents])
    monkeypatch.setattr(direct_sft, 'main', lambda _: None)
    out = tmp_path / 'out'
    rl_stage.main(['--rollouts', str(source), '--rl-tokenizer', str(tmp_path), '--out', str(out),
                  '--rl-held-text-data', str(held)])
    data = [json.loads(line) for line in (out / 'rl-text-data.jsonl').read_text().splitlines()]
    assert [r['text'] for r in data if r['split'] == 'test'] == ['protected']
    assert [r['split'] for r in data if r['text'] == 'task'] == ['train']
    receipt = json.loads((out / 'rl-data-receipt.json').read_text())
    assert receipt['held']['kind'] == 'ordinary-production-token-CE'
    assert 'rl_held_text_data' in recipe.HANDLERS['neuralese_rl']['optional_inputs']


def test_rft_epoch_exposes_each_distinct_trajectory_turn(monkeypatch):
    from natlang_neuralese.data import text_corpus
    from natlang_neuralese.train.direct_sft import direct_schedule, parse
    monkeypatch.setattr(text_corpus, 'tokenizer_fingerprint', lambda _: 'fp')
    monkeypatch.setattr(text_corpus, 'native_gold_packet', lambda tok, messages, target, tools, **kw:
                        (target['content'], [1, 2, 3], 1))
    documents = [{'task_id': 'same-task', 'split': 'train', 'messages': [], 'tools': None,
                  'target': {'content': 'turn-' + str(i)}} for i in range(7)]
    rows = rl_stage.render_rows(documents, object(), 'full')
    windows = [{'document': row['text'], 'sampling_group': row['sampling_group'],
                'cohort': 's7-rft', 'ids': [1, 2, 3]} for row in rows]
    a = parse(['--heads', 'h.pt', '--out', 'out', '--records', 'r.jsonl', '--epochs', '1', '--batch', '1'])
    _, plan = direct_schedule(windows, a)
    seen = {window['document'] for batch, *_ in plan for window in batch}
    assert seen == {document['target']['content'] for document in documents}


def test_rft_same_text_different_supervised_boundary_remains_distinct(monkeypatch):
    from natlang_neuralese.data import text_corpus
    monkeypatch.setattr(text_corpus, 'tokenizer_fingerprint', lambda _: 'fp')
    monkeypatch.setattr(text_corpus, 'native_gold_packet', lambda tok, messages, target, tools, **kw:
                        ('same-text', [1, 2, 3, 4], target['boundary']))
    rows = rl_stage.render_rows([{'task_id': 't', 'split': 'train', 'messages': [], 'tools': None,
         'target': {'boundary': boundary}} for boundary in [1, 2]], object(), 'full')
    assert rows[0]['text'] == rows[1]['text']
    assert rows[0]['sampling_group'] != rows[1]['sampling_group']


def test_value_rft_routes_to_shared_graph_ce_instead_of_text_flattening(tmp_path, monkeypatch):
    import transformers
    from natlang_neuralese.train import direct_sft
    from natlang_neuralese.data import text_corpus
    block = 'nz1_' + 'a' * 52
    rows = [_rollout('task', q) for q in [1., 0.]]
    for row in rows:
        row['turns'][0]['sampling'] = {'schema': 'natlang.rl-sampling/1', 'temperature': 1., 'guided': False,
            'prompt': [['tokens', [1]], ['block', block]], 'items': [['tok', 2]],
            'samples': [{'item': 0, 'token': 2, 'logp': -1.}]}
    source = tmp_path / 'rollouts.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    bundle = tmp_path / 'bundle.json'
    bundle.write_text(json.dumps({'blocks': {block: {'length': 2}}}))
    held = tmp_path / 'held.jsonl'
    held.write_text(json.dumps({'split': 'test', 'text': 'held', 'source_groups': ['h']}) + '\n')
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda path: object())
    monkeypatch.setattr(text_corpus, 'tokenizer_fingerprint', lambda _: 'fp')
    monkeypatch.setattr(rl_stage, 'render_rows', lambda *args: pytest.fail('must not flatten actual values'))
    calls = []
    monkeypatch.setattr(direct_sft, 'main', lambda argv: calls.append(argv))
    out = tmp_path / 'out'
    rl_stage.main(['--rollouts', str(source), '--rl-tokenizer', str(tmp_path), '--out', str(out),
        '--rl-mode', 'rft', '--rl-policy-id', 'policy', '--rl-block-bundle', str(bundle),
        '--rl-held-text-data', str(held)])
    spec = json.loads((out / 'rl-objective.json').read_text())
    assert spec['mode'] == 'rft' and spec['block_bundle'] == str(bundle)
    assert spec['documents'][0]['sampling']['prompt'][1] == ['block', block]
    assert '--rl-objective' in calls[0]


def test_value_document_detection_does_not_cross_tokenize_plain_teacher_turns():
    block = 'nz1_' + 'a' * 52
    ordinary = {'messages': [], 'target': {'content': 'ordinary teacher text'},
                'producers': [{'sampling': {'prompt': [['block', block]]}}]}
    assert not rl_stage.document_has_values(ordinary)
    assert rl_stage.document_has_values({**ordinary, 'sampling': {'prompt': [['block', block]]}})
