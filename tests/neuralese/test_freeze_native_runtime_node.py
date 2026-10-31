import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    'freeze_native_runtime', Path(__file__).parents[2] / 'scripts/neuralese_data/freeze_native_runtime.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_node_closure_is_copied_and_content_bound(tmp_path):
    source = tmp_path / 'source'
    package = source / 'just-bash'
    package.mkdir(parents=True)
    (package / 'package.json').write_text(json.dumps({'version': '3.4.2'}))
    (package / 'index.js').write_text('export default 1')
    first = module.freeze_node_dependencies(source, tmp_path / 'frozen')
    (package / 'index.js').write_text('export default 2')
    assert (tmp_path / 'frozen/just-bash/index.js').read_text() == 'export default 1'
    second = module.freeze_node_dependencies(source, tmp_path / 'next')
    assert first['sha256'] != second['sha256']
    assert first['just_bash_version'] == '3.4.2'


def test_node_closure_requires_collector_dependency(tmp_path):
    source = tmp_path / 'empty'
    source.mkdir()
    with pytest.raises(ValueError, match='just-bash'):
        module.freeze_node_dependencies(source, tmp_path / 'frozen')
