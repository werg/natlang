import pytest

from natlang_neuralese.serve.__main__ import main


def test_server_cli_no_longer_advertises_projection_loading(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "--projection" not in capsys.readouterr().out



def test_legacy_adapter_capabilities_are_not_advertised():
    from natlang_neuralese.serve.http import CAPABILITIES
    assert not any(capability.startswith("adapters.") for capability in CAPABILITIES)


def test_legacy_adapter_request_is_rejected_before_runtime_execution():
    import json
    import threading
    import urllib.error
    import urllib.request
    from types import SimpleNamespace
    from natlang_neuralese.serve.http import serve

    server = serve(SimpleNamespace(native_state_enabled=False), "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        request = urllib.request.Request(endpoint + "/v1/neuralese/grad", data=json.dumps({"adapters": []}).encode(),
                                         headers={"content-type": "application/json"}, method="POST")
        with pytest.raises(urllib.error.HTTPError) as response:
            urllib.request.urlopen(request)
        assert response.value.code == 400
        assert json.load(response.value)["error"]["code"] == "bad-request-field"
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(endpoint + "/v1/neuralese/adapters/nz1_" + "a" * 52 + "/lora")
        assert missing.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
