import copy

import pytest

from natlang.codebase import from_definitions
from natlang.diag import Reject
from natlang.host import load_definitions
from natlang.runtime import Runtime


def sources():
    return {
        "main": {"description": "Apply a tax rate", "args": {"price": "Num"}, "returns": "Num",
                 "code": "return args.price * 1.2;", "uses": {"helper": "helper"}},
        "helper": {"args": {"value": "Num"}, "returns": "Num", "code": "return args.value + 1;"},
    }


def test_in_memory_source_graph_is_checked_and_snapshotted():
    supplied = sources()
    graph, root = load_definitions(supplied, "main", {"price": 10})
    assert graph.root.codebase["helper"] is graph.get("helper")
    revision = graph.revision
    supplied["main"]["code"] = "return 999;"
    assert graph.revision == revision and graph.root.body == "return args.price * 1.2;\n"
    out, value = Runtime(None).run_root(root)
    assert (out.kind, value) == ("done", 12)
    assert from_definitions(copy.deepcopy(sources()), "main").revision == revision


def test_missing_and_cyclic_links_fail_before_execution():
    missing = sources()
    missing["main"]["uses"] = {"helper": "unknown"}
    with pytest.raises(Reject):
        from_definitions(missing, "main")
    cyclic = sources()
    cyclic["helper"]["uses"] = {"main": "main"}
    with pytest.raises(Reject):
        from_definitions(cyclic, "main")
    invalid = sources()
    invalid["unused"] = {"args": {"x": "Imaginary"}, "returns": "Num", "code": "return 1;"}
    with pytest.raises(Reject):
        from_definitions(invalid, "main")


def test_input_text_is_value_even_if_it_looks_like_a_path(tmp_path):
    file = tmp_path / "secret"
    file.write_text("file contents")
    entries = {"main": {"args": {"text": "Text"}, "returns": "Text", "code": "return args.text;"}}
    _, root = load_definitions(entries, "main", {"text": str(file)})
    out, value = Runtime(None).run_root(root)
    assert out.kind == "done" and value == str(file)
