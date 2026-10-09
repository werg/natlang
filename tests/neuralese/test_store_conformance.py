"""Conformance of owner-scoped block holds, pins and collection (serve/store.py `TensorStore`, the fork's
`tensor_store`): the reference server and the llama.cpp fork answer the same owner-tagged requests
(`x-natlang-owner`) with the same removals. Uses the server pair of test_server_conformance.py (skipped unless the
fork's CPU build and the base model exist). The WebAssembly service has no request headers and stays single-owner.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

torch = pytest.importorskip("torch")

from test_server_conformance import _servers  # noqa: E402  (tests/neuralese is on sys.path under pytest)


@pytest.fixture(scope="module")
def servers(loaded, tmp_path_factory):
    pair = _servers(loaded, tmp_path_factory, "final")
    yield pair
    pair["process"].terminate()
    pair["server"].shutdown()
    pair["engine"].stop()


def _call(url, method, body=None, owner=None):
    data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode()
    headers = {"content-type": "application/octet-stream" if isinstance(body, bytes) else "application/json"}
    if owner:
        headers["x-natlang-owner"] = owner
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null")


def _scenario(url):
    """The owner protocol on one server; returns what a client observes."""
    from natlang_neuralese.serve.store import encode_block, make_block

    info = _call(url + "/v1/neuralese/info", "GET")[1]
    dialect, width = info["dialects"][0], info["width"]
    mine, theirs, anonymous = (make_block(torch.full((2, width), float(i)), dialect) for i in (1, 2, 3))
    seen = {"store": info.get("store")}
    for block, owner in ((mine, "s1"), (theirs, "s2"), (anonymous, None)):
        assert _call(f"{url}/v1/neuralese/blocks/{block.id}", "PUT", encode_block(block), owner)[0] == 201
    encoded = _call(url + "/v1/neuralese/encode", "POST", {"text": "owned by s1"}, "s1")[1]["id"]
    seen["anonymous collect"] = _call(url + "/v1/neuralese/collect", "POST", {"referenced": []})[1]["removed"]
    seen["s1 collect"] = _call(url + "/v1/neuralese/collect", "POST", {"referenced": []}, "s1")[1]["removed"]
    seen["s2 pin"] = _call(f"{url}/v1/neuralese/blocks/{theirs.id}/pin", "POST", None, "s2")[0]
    seen["s2 collect while pinned"] = _call(url + "/v1/neuralese/collect", "POST", {"referenced": []}, "s2")[1]["removed"]
    seen["s1 cannot unpin"] = _call(f"{url}/v1/neuralese/blocks/{theirs.id}/unpin", "POST", None, "s1")[0]
    seen["s2 collect, s1 unpinned"] = _call(url + "/v1/neuralese/collect", "POST", {"referenced": []}, "s2")[1]["removed"]
    _call(f"{url}/v1/neuralese/blocks/{theirs.id}/unpin", "POST", None, "s2")
    seen["s2 collect after unpin"] = _call(url + "/v1/neuralese/collect", "POST", {"referenced": []}, "s2")[1]["removed"]
    names = {mine.id: "mine", theirs.id: "theirs", anonymous.id: "anonymous", encoded: "encoded"}
    return {key: sorted(names.get(i, i) for i in value) if isinstance(value, list) else value for key, value in seen.items()}


def test_owner_scoped_collection_agrees(servers):
    reference, fork = _scenario(servers["reference"]), _scenario(servers["fork"])
    assert reference == {"store": {"owners": True, "persistent": False}, "anonymous collect": ["anonymous"],
                         "s1 collect": ["encoded", "mine"], "s2 pin": 200, "s2 collect while pinned": [],
                         "s1 cannot unpin": 200, "s2 collect, s1 unpinned": [], "s2 collect after unpin": ["theirs"]}
    assert fork == reference
