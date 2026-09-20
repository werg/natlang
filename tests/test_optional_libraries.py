from hosts.model_help import HelpRequest, ModelHelp
from hosts.rendering import DocumentRenderer
from hosts.search import ExactSearch


def test_exact_search_finds_supplied_evidence_with_provenance():
    search = ExactSearch({"incident-1": "09:01 started\n09:02 timeout in build job",
                          "incident-2": "09:03 timeout cleared"})
    hits = search.scan("timeout")
    assert [(h.document, h.line) for h in hits] == [("incident-1", 2), ("incident-2", 1)]
    assert search.lookup("incident-1").startswith("09:01")


def test_model_help_reports_usage_and_failure_explicitly():
    request = HelpRequest("fixture-vision", "Describe the supplied frame", max_tokens=20, seed=4)
    helper = ModelHelp(lambda req: {"observation": "A red frame", "usage": {"completion_tokens": 4}})
    result = helper.request(request)
    assert result.status == "ok" and result.usage["completion_tokens"] == 4
    failed = ModelHelp(lambda req: (_ for _ in ()).throw(TimeoutError("late"))).request(request)
    assert failed.status == "failed" and "late" in failed.error


def test_typed_document_renders_two_outputs_and_records_ui_event():
    renderer = DocumentRenderer()
    blocks = [{"kind": "heading", "text": "Plan & notes"},
              {"kind": "paragraph", "text": "Ship <today>."}]
    assert renderer.render(blocks, "html") == "<h1>Plan &amp; notes</h1>\n<p>Ship &lt;today&gt;.</p>"
    assert renderer.render(blocks, "text") == "Plan & notes\nShip <today>."
    renderer.emit("clicked", {"id": "publish"})
    assert [e["kind"] for e in renderer.events] == ["render", "render", "ui"]
