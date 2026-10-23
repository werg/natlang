import pytest

from natlang_neuralese.serve.__main__ import main
from natlang_neuralese.serve.engine import Engine
from natlang_neuralese.serve.chat import RequestError


def test_server_cli_no_longer_advertises_projection_loading(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "--projection" not in capsys.readouterr().out


def test_projected_adapter_code_is_rejected_in_favor_of_native_state():
    engine = object.__new__(Engine)
    with pytest.raises(RequestError, match="neuralese-native-state-required"):
        engine.resolve_adapters([{"code": "nz1_" + "a" * 26, "projection": "retired"}])
