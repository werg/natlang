"""Queue transport cannot replace the controller's real full continuation check."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
from natlang_neuralese.rl import native_bridge as bridge, native_rounds, production
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.loop import capture_training_rng_state
from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA


def fixture(tmp_path, monkeypatch):
    parent = tmp_path/'parent.pt'; parent.write_bytes(b'exact original actor')
    out = tmp_path/'production'; config = tmp_path/'config.json'
    write_json_durable(config, {'native_state':True, 'checkpoint':str(parent), 'out':str(out)})
    request = bridge.write_request(tmp_path/'requests', kind='round', out=out, config=config)
    checkpoint = out/'learner/checkpoint.pt'
    def run(*args, **kwargs):
        checkpoint.parent.mkdir()
        torch.save({'schema':CHECKPOINT_SCHEMA, 'training':{'schema':TRAINING_SCHEMA,
            'optimizer':{'actual-moment':torch.ones(1)}, 'rng':capture_training_rng_state('cpu'),
            'options':{'continuation':{'checkpoint_sha256':bridge.binding(parent)['sha256']}}}}, checkpoint)
        state = {'checkpoint':str(checkpoint), 'phase':'collect'}
        write_json_durable(out/'state.json', state)
        return state
    monkeypatch.setattr(production, 'run', run)
    result = request.with_suffix('.result.json')
    assert bridge.execute(request, result)['status'] == 'completed'
    loads = []
    original_load = CheckpointPolicy.load
    def load(*args, **kwargs):
        loads.append(str(args[0]))
        return original_load(*args, **kwargs)
    monkeypatch.setattr(CheckpointPolicy, 'load', load)
    return request, result, checkpoint, loads


def command(tmp_path):
    return [sys.executable, '-m', 'natlang_neuralese.rl.native_bridge', '--queue', str(tmp_path/'queue'),
        '--request', '{request}', '--result', '{result}', '--stop-file', '{stop_file}']


def old_cli_environment(tmp_path):
    """Real child import path whose historical bridge rejects any invocation."""
    root=tmp_path/'historical-pythonpath'
    from natlang_neuralese.common import hashing
    actual=Path(hashing.__file__).resolve().parents[1]
    package=root/'natlang_neuralese';package.mkdir(parents=True)
    (package/'__init__.py').write_text('__path__.append('+repr(str(actual))+')\n')
    rl=package/'rl';rl.mkdir()
    (rl/'__init__.py').write_text('__path__.append('+repr(str(actual/'rl'))+')\n')
    marker=tmp_path/'historical-cli-executed'
    (rl/'native_bridge.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').touch()\nraise RuntimeError("historical CLI has no deferred flag")\n')
    return {**os.environ,'PYTHONPATH':str(root)},marker


@pytest.mark.parametrize('deferred', [False, True])
def test_standalone_default_still_decodes_and_explicit_transport_defers_only_payload(tmp_path, monkeypatch, deferred):
    request, result, _, loads = fixture(tmp_path, monkeypatch)
    value = bridge.enqueue(request, result, tmp_path/'queue', tmp_path/'stop.json',
        stop=SimpleNamespace(requested=False), defer_checkpoint_verification=deferred)
    assert value['status'] == 'completed' and len(loads) == (0 if deferred else 1)
    assert bridge.verify_result(request, result) == value
    assert len(loads) == (1 if deferred else 2)


def test_actual_queue_cli_then_controller_performs_exactly_one_full_decode(tmp_path, monkeypatch):
    request, result, checkpoint, loads = fixture(tmp_path, monkeypatch)
    argv = command(tmp_path); original = list(argv)
    environment,old_cli=old_cli_environment(tmp_path)
    def run(template, values):
        assert template[-1] == '--defer-checkpoint-verification'
        assert template[1]=='-c'and template[3]==bridge.__file__
        parts = [part.format(**{key:str(value) for key,value in values.items()}) for part in template]
        child=subprocess.run(parts,env=environment,capture_output=True,text=True)
        assert child.returncode==0,child.stderr
        assert not old_cli.exists()
        assert loads == []
    monkeypatch.setattr(native_rounds, 'run_command', run)
    value = native_rounds.bridge_result(argv, request, SimpleNamespace(requested=False))
    assert value['checkpoint'] == bridge.binding(checkpoint)
    assert loads == [str(checkpoint)] and argv == original


def test_copied_current_module_bootstrap_checks_exact_source_before_execution(tmp_path,monkeypatch):
    request,result,_,loads=fixture(tmp_path,monkeypatch)
    source=tmp_path/'published-current-bridge.py'
    source.write_bytes(Path(bridge.__file__).read_bytes())
    name='natlang_neuralese.rl.copied_queue_bridge'
    spec=importlib.util.spec_from_file_location(name,source)
    current=importlib.util.module_from_spec(spec);spec.loader.exec_module(current)
    environment,old_cli=old_cli_environment(tmp_path)
    argv=current.queue_client_command(command(tmp_path))
    assert argv[3]==str(source)and argv.count('--defer-checkpoint-verification')==1
    values={'request':str(request),'result':str(result),'stop_file':str(tmp_path/'stop.json')}
    actual=[part.format(**values)for part in argv]
    child=subprocess.run(actual,env=environment,capture_output=True,text=True)
    assert child.returncode==0,child.stderr
    assert not old_cli.exists()and loads==[]
    source.write_bytes(source.read_bytes()+b'\n# changed after command binding\n')
    rejected=subprocess.run(actual,env=environment,capture_output=True,text=True)
    assert rejected.returncode!=0 and 'helper source binding changed'in rejected.stderr
    assert not old_cli.exists()and loads==[]


@pytest.mark.parametrize('tamper', ['bytes', 'parent', 'rng', 'optimizer', 'pruned'])
def test_deferred_client_never_grants_bad_artifact_or_continuation(tmp_path, monkeypatch, tamper):
    request, result, checkpoint, loads = fixture(tmp_path, monkeypatch)
    if tamper == 'bytes':
        checkpoint.write_bytes(b'changed after result publication')
    else:
        saved = torch.load(checkpoint, weights_only=True)
        if tamper == 'parent': saved['training']['options']['continuation']['checkpoint_sha256'] = 'f'*64
        elif tamper == 'pruned': saved['optimizer_pruned'] = True
        else: saved['training'].pop(tamper)
        torch.save(saved, checkpoint)
        value = json.loads(result.read_text())
        value.update(checkpoint=bridge.binding(checkpoint), artifacts=bridge.inventory(checkpoint.parents[1]))
        write_json_durable(result, value)
    def run(template, values):
        assert template[-1] == '--defer-checkpoint-verification'
        assert bridge.enqueue(request, result, tmp_path/'queue', tmp_path/'stop.json',
            stop=SimpleNamespace(requested=False), defer_checkpoint_verification=True)['status'] == 'completed'
        assert loads == []
    monkeypatch.setattr(native_rounds, 'run_command', run)
    with pytest.raises(ValueError):
        native_rounds.bridge_result(command(tmp_path), request, SimpleNamespace(requested=False))


def test_custom_provider_command_unchanged_and_deferred_nonqueue_cli_rejects(tmp_path, monkeypatch):
    request, result, checkpoint, loads = fixture(tmp_path, monkeypatch)
    argv = ['custom-provider', '--request', '{request}']
    def run(actual, values): assert actual == argv
    monkeypatch.setattr(native_rounds, 'run_command', run)
    native_rounds.bridge_result(argv, request, SimpleNamespace(requested=False))
    assert loads == [str(checkpoint)]
    with pytest.raises(SystemExit) as error:
        bridge.main(['--request',str(request),'--result',str(result),'--defer-checkpoint-verification'])
    assert error.value.code == 2
