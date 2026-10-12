from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full_check_builds_node_collector_before_python_suite():
    script = (ROOT / 'scripts/check-main.sh').read_text()
    node_build = script.index("step ts-build-node sh -c 'cd ts-host && npm run --silent build:node'")
    python_suite = script.index('step python-tests env HF_HOME=')

    assert node_build < python_suite
    assert 'if [ $quick = 0 ] && { [ $python = 1 ] || [ $ts = 1 ]; }; then' in script
    assert 'node_build_done=1' in script
