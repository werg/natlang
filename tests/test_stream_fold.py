import pytest

from natlang.runtime import Runtime
from natlang.streams import QueueSource, StreamBuffer, WindowedMap
from natlang.values import load_program


def fold(source):
    root = load_program({"$fold": {"type": "Fold<Text, Num>", "init": 0,
        "step": {"$lambda": {"type": "Lambda<{ acc: Num, item: Text }, Num>",
                            "code": "return args.acc + args.item.length;"}}}})
    root.over = StreamBuffer(source)
    return root


def test_empty_waits_close_drains_and_close_token_is_data():
    source = QueueSource(capacity=2)
    root = fold(source)
    rt = Runtime(None)
    out, value = rt.run_root(root)
    assert out.kind == "waiting" and root.status == "waiting"
    source.put("$close")
    out, value = rt.run_root(root)
    assert out.kind == "waiting" and root.acc == 6 and source.poll().kind == "empty"
    assert root.over.position == 1 and root.over.current is None
    source.close()
    out, value = rt.run_root(root)
    assert (out.kind, value) == ("done", 6)


def test_source_failure_is_distinct_and_queue_is_bounded():
    source = QueueSource(capacity=1)
    source.put("a")
    with pytest.raises(BufferError):
        source.put("b")
    source.fail("producer disconnected")
    out, root = Runtime(None).run_root(fold(source))
    assert out.kind == "quiesced" and "producer disconnected" in out.detail
    assert root.acc == 1


def test_long_stream_releases_consumed_payloads():
    source = QueueSource(capacity=2)
    root = fold(source)
    rt = Runtime(None)
    for i in range(100):
        source.put("x")
        out, _ = rt.run_root(root)
        assert out.kind == "waiting" and not source.items
        assert root.over.current is None and root.over.position == i + 1
    source.close()
    out, value = rt.run_root(root)
    assert (out.kind, value) == ("done", 100)


def test_windowed_map_forwards_ordered_results_without_infinite_list():
    source = QueueSource(capacity=8)
    for item in range(7):
        source.put(item)
    source.close()
    mapped = WindowedMap(source, lambda items: [item * 2 for item in items], width=3)
    results = []
    while (polled := mapped.poll()).kind == "item":
        results.append(polled.value)
        assert len(mapped.ready) <= 2
    assert polled.kind == "closed" and results == [i * 2 for i in range(7)]
