from scripts.check_runtime_parity import violations


def test_runtime_change_requires_a_paired_fixture_and_counterpart():
    assert len(violations(["natlang/runtime.py"])) == 2
    assert violations(["natlang/runtime.py", "ts-host/src/native/runtime.ts"])
    assert violations(["natlang/runtime.py", "ts-host/src/native/runtime.ts",
                       "ts-host/test/native-parity.test.mjs"]) == []


def test_typescript_fix_can_use_a_python_comparison_fixture():
    assert violations(["ts-host/src/native/runtime.ts", "ts-host/test/native-parity.test.mjs"]) == []
    assert violations(["ts-host/src/native/runtime.ts"]) != []


def test_browser_only_change_does_not_trigger_cross_runtime_gate():
    assert violations(["ts-host/src/browser/local-model.ts"]) == []
