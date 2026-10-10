"""The view/ask corpus builder (scripts/neuralese_data/view_corpus.py) on small local fixtures; no downloads."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from neuralese_data import view_corpus as vc  # noqa: E402
from neuralese_data import view_extract as vx  # noqa: E402
from neuralese_data.records import validate  # noqa: E402

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

ARTICLE = ("The river town of Aldmere held its first lantern festival in 1952. " * 6).strip()
FUNC = '''def load_config(path, strict=False, *extra, **options):
    """Read a configuration file and return its settings as a dictionary.

    Longer explanation that is not part of the summary.
    """
    text = open(path).read()
    data = json.loads(text)
    if strict:
        validate_schema(data)
    return merge_defaults(data, options)
'''
HTML = """<html><head><title>Aldmere Cars</title><script>var x = 1;</script></head>
<body><h2 tid="3">Specs</h2><h2 tid="4">Prices</h2>
<div id="price-main" tid="5">Base price: $21,000</div>
<table><tr><th>Model</th><th>Doors</th></tr><tr><td>Alda</td><td>4</td></tr></table>
<a href="https://example.org/alda">Alda</a> <a href="/compare">Compare</a>
<p>""" + "Filler text about cars. " * 20 + "</p></body></html>"""
BASH = ("============================= test session starts ==============================\n"
        "collected 3 items\n\ntests/test_a.py .F.\n\n"
        "Traceback (most recent call last):\n  File \"/workspace/project/src/mod.py\", line 12, in run\n"
        "    raise ValueError('bad input')\nValueError: bad input\n"
        "FAILED tests/test_a.py::test_two - ValueError: bad input\n"
        "========================= 1 failed, 2 passed in 0.12s =========================\n"
        + "log line\n" * 120 + "[Command finished with exit code 1]")


# ------------------------------------------------------------------------------------------- extraction engines

def test_json_paths_and_extracts_are_computed():
    value = {"name": "pkg", "scripts": {"build": "tsc", "test": "jest --ci"}, "files": ["a", "b", "c"],
             "weird key": {"x y": 1}}
    paths = dict(vx.walk(value))
    assert paths["$.scripts.build"] == "tsc"
    assert paths["$.files[2]"] == "c"
    assert paths['$["weird key"]["x y"]'] == 1
    extracts = vc._json_extracts(value, vc.random.Random(0), limit=4)
    assert {m for _, _, m in extracts} == {"jsonpath-value", "jsonpath-keys", "jsonpath-length", "jsonpath-locate"}
    for request, answer, method in extracts:
        if method == "jsonpath-length":
            path = request.split("`")[1]
            assert answer == str(len(paths[path]))
        if method == "jsonpath-locate":
            assert answer in paths


def test_css_selector_subset():
    dom = vx.HtmlDoc(vx.clean_html(HTML, drop_attrs=("tid",)))
    assert dom.title() == "Aldmere Cars"
    assert [n.text() for n in dom.select("h2")] == ["Specs", "Prices"]
    assert dom.select("#price-main")[0].text() == "Base price: $21,000"
    assert [n.attrs["href"] for n in dom.select("a[href]")] == ["https://example.org/alda", "/compare"]
    assert len(dom.select("table tr")) == 2 and len(dom.select("table > tr")) == 2
    assert len(dom.select("body td")) == 2
    cleaned = vx.clean_html(HTML, drop_attrs=("tid",))
    assert "tid=" not in cleaned and "var x" not in cleaned


def test_python_facts_and_docstring_removal():
    facts = vx.python_facts(FUNC)
    assert facts["name"] == "load_config"
    assert facts["params"] == ["path", "strict", "*extra", "**options"]
    # A method called on an expression (`open(path).read()`) is listed by its attribute chain.
    assert facts["calls"] == ["open", "read", "json.loads", "validate_schema", "merge_defaults"]
    code = vx.without_docstring(facts)
    assert "Read a configuration" not in code and "merge_defaults" in code
    compile(code, "<fixture>", "exec")
    assert vx.summary_sentence(facts["docstring"]) == "Read a configuration file and return its settings as a dictionary."


def test_tool_output_facts():
    assert vx.exit_code(BASH) == 1
    assert vx.failing_tests(BASH) == ["tests/test_a.py::test_two"]
    assert vx.pytest_summary(BASH) == "1 failed, 2 passed"
    exc = vx.final_exception(BASH)
    assert (exc["type"], exc["message"], exc["path"], exc["line"]) == ("ValueError", "bad input", "/workspace/project/src/mod.py", 12)


def test_table_extracts():
    rows = [["Team", "Wins"], ["Ash", "3"], ["Birch", "7"], ["Cedar", "5"]]
    out = {m: (q, a) for q, a, m in vc._table_extracts(rows, vc.random.Random(1), limit=3)}
    assert out["table-row-count"][1] == "3"
    assert out["table-argmax"][1] == "Birch"


def test_qmsum_windows_pack_and_pad():
    turns = [f"A: turn {i} " + "x" * 90 for i in range(400)]
    windows = vc.qmsum_windows(turns, [(10, 12), (14, 15), (300, 302)], cap=20_000)
    assert [w[2] for w in windows] == [[0, 1], [2]]
    a, b, _ = windows[0]
    assert a <= 10 and b >= 15 and sum(len(t) + 1 for t in turns[a:b + 1]) >= 6000
    # A short meeting: both windows pad to the whole transcript and become one source with two purposes.
    short = turns[:30]
    assert vc.qmsum_windows(short, [(2, 3), (25, 26)], cap=4_000) == [(0, 29, [0, 1])]


# ------------------------------------------------------------------------------------------- end to end

def _parquet(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), str(path))


def _git(repo: Path, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"})


@pytest.fixture()
def fixtures(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    hf = tmp_path / "hf"
    cnn = hf / "datasets--cnn_dailymail/snapshots/rev1/3.0.0"
    for split, name in (("train", "train-00000-of-00003"), ("validation", "validation-00000-of-00001"), ("test", "test-00000-of-00001")):
        _parquet(cnn / f"{name}.parquet", [{"article": f"{ARTICLE} Story {split} {i}.", "highlights": f"Aldmere held a festival .\nStory {i} .",
                                           "id": f"{split}{i:03d}"} for i in range(6)])
    squad = hf / "datasets--rajpurkar--squad/snapshots/rev2/plain_text/train-00000-of-00001.parquet"
    _parquet(squad, [{"id": f"q{i}", "title": "Aldmere", "context": ARTICLE, "question": f"When did Aldmere first hold it ({i})?",
                      "answers": {"text": ["1952", "in 1952"], "answer_start": [0, 0]}} for i in range(3)])
    quality = hf / "datasets--tasksource--quality/snapshots/rev3/data"
    for split in ("train", "validation"):
        _parquet(quality / f"{split}-00000.parquet", [{"article_id": f"{split}-a", "title": "Lanterns", "article": ARTICLE,
                                                      "question": "What was held?", "question_unique_id": f"{split}-a-1",
                                                      "options": "['A fair', 'A lantern festival', 'A race', 'A vote']",
                                                      "gold_label": "2", "license": "Public domain"}])
    xlam = tmp_path / "xlam.json"
    tools = json.dumps([{"name": "get_weather", "description": "Weather for a city, by name and unit system.",
                         "parameters": {"city": {"type": "str", "description": "City name"},
                                        "units": {"type": "str", "default": "metric", "description": "Unit system"}}},
                        {"name": "get_time", "description": "Local time for a timezone identifier string.",
                         "parameters": {"tz": {"type": "str", "description": "IANA timezone"}}}])
    xlam.write_text(json.dumps([{"id": 0, "query": "Weather in Oslo?", "tools": tools,
                                 "answers": json.dumps([{"name": "get_weather", "arguments": {"city": "Oslo"}}])}]))
    repos = tmp_path / "repos"
    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "-q")
    (work / "LICENSE").write_text("MIT License\n\nPermission is hereby granted, free of charge, to any person ...\n")
    (work / "package.json").write_text(json.dumps({"name": "demo", "version": "1.2.3", "scripts": {"build": "tsc -p .", "test": "jest"},
                                                   "files": ["dist", "src", "README.md"]}, indent=2) + "\n")
    (work / ".github").mkdir()
    (work / ".github/ci.yml").write_text("name: ci\non: [push]\njobs:\n  test:\n    runs-on: ubuntu-latest\n"
                                         "    steps:\n      - uses: actions/checkout@v4\n      - run: npm test\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "init")
    (repos / "octo").mkdir(parents=True)
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(repos / "octo" / "demo")], check=True, capture_output=True)
    nebius = tmp_path / "traj.parquet"
    trajectory = [
        {"role": "system", "content": "sys", "name": None, "tool_call_id": None, "tool_calls": None},
        {"role": "user", "content": "<uploaded_files>\n/workspace/demo__1.0\n</uploaded_files>\nFix the bug.", "name": None,
         "tool_call_id": None, "tool_calls": None},
        {"role": "assistant", "content": "", "name": None, "tool_call_id": None,
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "execute_bash", "arguments": json.dumps({"command": "pytest -q"})}}]},
        {"role": "tool", "content": BASH, "name": "execute_bash", "tool_call_id": "c1", "tool_calls": None},
    ]
    _parquet(nebius, [{"trajectory_id": f"t{i}", "instance_id": f"octo__demo-{i}", "repo": "octo/demo", "trajectory": trajectory}
                      for i in range(16)])
    # Fetched sources.
    (raw / "qmsum/data/ALL/jsonl").mkdir(parents=True)
    meeting = {"meeting_transcripts": [{"speaker": "A" if i % 2 else "B", "content": f"Point {i} about the budget. " * 4} for i in range(60)],
               "general_query_list": [{"query": "Summarize the meeting.", "answer": "They discussed the budget."}],
               "specific_query_list": [{"query": "What did they say about point 5?", "answer": "Point 5 concerned the budget.",
                                        "relevant_text_span": [["5", "6"]]},
                                       {"query": "And point 8?", "answer": "Point 8 also concerned the budget.",
                                        "relevant_text_span": [["8", "8"]]}],
               "topic_list": []}
    for name in ("train", "val", "test"):
        (raw / f"qmsum/data/ALL/jsonl/{name}.jsonl").write_text(json.dumps(meeting) + "\n")
    (raw / "fetaqa/data").mkdir(parents=True)
    feta = {"feta_id": 7, "table_page_title": "Aldmere", "table_section_title": "Festivals", "page_wikipedia_url": "https://en.wikipedia.org/wiki/Aldmere",
            "table_array": [["Year", "Visitors"], ["1952", "300"], ["1953", "450"], ["1954", "400"]] + [[str(1955 + i), str(500 + i)] for i in range(10)],
            "question": "How many visitors came in 1953?", "answer": "In 1953, 450 visitors came."}
    for name in ("train", "dev", "test"):
        (raw / f"fetaqa/data/fetaQA-v1_{name}.jsonl").write_text(json.dumps(feta | {"feta_id": f"{name}7"}) + "\n")
    (raw / "wtq/repo/data").mkdir(parents=True)
    (raw / "wtq/repo/csv/204-csv").mkdir(parents=True)
    (raw / "wtq/repo/csv/204-csv/1.csv").write_text('"Team","Wins"\n' + "".join(f'"T{i}","{i * 3 % 17}"\n' for i in range(12)))
    for fname in ("training.tsv", "pristine-unseen-tables.tsv"):
        (raw / f"wtq/repo/data/{fname}").write_text("id\tutterance\tcontext\ttargetValue\n"
                                                    f"nt-{fname[0]}1\twhich team won most?\tcsv/204-csv/1.csv\tT11\n"
                                                    f"nt-{fname[0]}2\thow many teams?\tcsv/204-csv/1.csv\t12\n")
    _parquet(raw / "codesearchnet/python-train.parquet", [{"repository_name": "octo/demo", "func_path_in_repository": "cfg.py",
                                                           "func_name": "load_config", "whole_func_string": FUNC,
                                                           "func_code_url": "https://github.com/octo/demo/blob/x/cfg.py#L1"}])
    for split in ("validation", "test"):
        _parquet(raw / f"codesearchnet/python-{split}.parquet", [{"repository_name": f"octo/{split}", "func_path_in_repository": "cfg.py",
                                                                   "func_name": "load_config", "whole_func_string": FUNC,
                                                                   "func_code_url": "u"}])
    site = raw / "websrc/release/auto/01"
    (site / "processed_data").mkdir(parents=True)
    (site / "processed_data/0100001.html").write_text(HTML)
    (site / "dataset.csv").write_text("question,id,element_id,answer_start,answer\n"
                                      "What is the base price?,au010000100001,5,12,\"$21,000\"\n"
                                      "How many doors does the Alda have?,au010000100002,9,0,4\n")
    sources = {"files": [{"dataset": d, "revision": f"rev-{d}"} for d in ("qmsum", "fetaqa", "wtq", "codesearchnet", "websrc")]}
    (raw / "sources.json").write_text(json.dumps(sources))
    monkeypatch.setattr(vc, "HF", hf)
    monkeypatch.setattr(vc, "XLAM", xlam)
    monkeypatch.setattr(vc, "REPOS", repos)
    monkeypatch.setattr(vc, "NEBIUS", nebius)
    return raw


def _records(out: Path) -> list[dict]:
    return [json.loads(l) for p in sorted(out.glob("*.port-records.jsonl")) for l in p.read_text().splitlines() if l.strip()]


def test_build_end_to_end(fixtures, tmp_path):
    out = tmp_path / "out"
    caps = {k: 4 for k in vc.DEFAULT_CAPS}
    manifest = vc.build(fixtures, out, corpus_id="fixture", caps=caps, min_source_chars=50, protected_path=None, log=lambda *_: None)
    records = _records(out)
    assert manifest["records"] == len(records) > 0
    assert set(manifest["by_artifact"]) == {"prose", "code", "html", "data", "table", "log"}
    assert {r["task"] for r in records} == {"reconstruct", "consume", "compare"}
    assert manifest["admission"]["training_admission"] is False
    for r in records:
        assert validate(r) == [], (r["id"], validate(r))
        assert r["writer"]["result_type"] == "Neuralese<string>"
        assert r["family"] == f"view_{r['lineage']['notes']['artifact']}_{r['id'].split(':')[-2]}"
        assert r["lineage"]["teacher"] is None and r["outcome"]["label"] == "gold"
        if r["task"] == "reconstruct":
            assert r["target"]["value"] == r["sources"][0]["text"]
        else:  # a record's own target is never handed over as an exact ref
            target = r["target"]["value"] if isinstance(r["target"]["value"], str) else json.dumps(r["target"]["value"])
            assert all(ref["text"] not in target for ref in r["sources"][0]["exact_refs"])
    by_id = {r["id"]: r for r in records}
    for r in records:
        if r["task"] == "compare":
            assert r["contrasts"]["purpose_pairs"]
            for other in r["contrasts"]["purpose_pairs"]:
                assert by_id[other]["sources"][0]["text"] == r["sources"][0]["text"]
    # Splits are closed over groups.
    seen = {}
    for r in records:
        for g in r["split_groups"]:
            assert seen.setdefault(g, r["split"]) == r["split"]
    # Code: the docstring summary never appears in its source.
    code = [r for r in records if r["family"] == "view_code_summary"]
    assert code and all(r["target"]["value"] not in r["sources"][0]["text"] for r in code)
    # Computed targets agree with recomputation.
    tables = [r for r in records if r["lineage"]["notes"].get("extractor") == "table-row-count"]
    assert tables and all(r["target"]["value"] == str(r["sources"][0]["text"].count("\n")) for r in tables)
    expected = {"log-exit-code": "1", "log-failing-tests": "tests/test_a.py::test_two", "log-pytest-summary": "1 failed, 2 passed",
                "log-exception": "ValueError: bad input", "log-innermost-frame": "/workspace/project/src/mod.py:12"}
    logs = [r for r in records if r["family"] == "view_log_extract"]
    assert logs and all(r["target"]["value"] == expected[r["lineage"]["notes"]["extractor"]] for r in logs)
    # The exit code only fills a slot nothing more informative takes (here: three richer facts exist).
    assert not any(r["lineage"]["notes"]["extractor"] == "log-exit-code" for r in logs)
    data = [r for r in records if r["lineage"]["store"] == "repo_configs"]
    assert data and {r["license"]["spdx"] for r in data} == {"MIT"}
    yaml = pytest.importorskip("yaml")
    for r in data:
        if r["lineage"]["notes"].get("extractor") in ("jsonpath-length", "jsonpath-value"):
            paths = dict(vx.walk(yaml.safe_load(r["sources"][0]["text"])))
            node = paths[r["consumer"]["context"][0]["content"].split("`")[1]]
            expected = str(len(node)) if isinstance(node, list) else vx.scalar_text(node)
            assert r["target"]["value"] == expected


def test_build_is_deterministic(fixtures, tmp_path):
    caps = {k: 4 for k in vc.DEFAULT_CAPS}
    a = vc.build(fixtures, tmp_path / "a", corpus_id="x", caps=caps, min_source_chars=50, protected_path=None, log=lambda *_: None)
    b = vc.build(fixtures, tmp_path / "b", corpus_id="x", caps=caps, min_source_chars=50, protected_path=None, log=lambda *_: None)
    assert [f["sha256"] for f in a["files"]] == [f["sha256"] for f in b["files"]]


def test_protected_hit_forces_test(fixtures, tmp_path):
    from neuralese_data.records import text_hash

    protected = tmp_path / "protected.json"
    protected.write_text(json.dumps({"question_hashes": {text_hash("What was held?"): "bench"}, "ids": {}}))
    vc.build(fixtures, tmp_path / "p", corpus_id="p", caps={k: 4 for k in vc.DEFAULT_CAPS}, min_source_chars=50,
             protected_path=protected, only=["quality"], log=lambda *_: None)
    records = _records(tmp_path / "p")
    assert records and all(r["split"] == "test" for r in records)
