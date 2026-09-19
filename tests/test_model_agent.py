"""The model-backed agent loop, driven by a scripted decoder.

The fake decoder checks that every scripted output is accepted by the grammar
it was given, so this exercises two-phase decoding end to end without a model.
"""
import yaml

from natlang import gbnf
from natlang.agents import OracleAgent
from natlang.decoder import Generation
from natlang.model_agent import ModelAgent
from natlang.runtime import Runtime

from test_canonical_traces import PROGRAMS, oracle, parse_trace
from test_grammar import _root


class ScriptedDecoder:
    def __init__(self, outputs):
        self.outputs, self.calls, self.violations = list(outputs), [], []

    def format(self, messages):
        return "".join(f"<{m['role']}>\n{m['content']}\n" for m in messages) + "<assistant>\n"

    def generate(self, prompt, *, grammar, max_tokens, temperature, seed, stop, n_probs=0):
        text = self.outputs.pop(0)
        self.calls.append((prompt, grammar, text))
        if grammar:
            candidate = text if text.endswith("\n") or text == "" else text + "\n"
            if not (gbnf.accepts(grammar, text) or gbnf.accepts(grammar, candidate)):
                self.violations.append(text)
        return Generation(text)


def _script(actions):
    """Split actions into the separate decoder outputs the agent will request."""
    outs = []
    for a in actions:
        header, _, body = a.partition("\n")
        outs.append(header + "\n")
        if header.split()[0] in ("set", "edit", "eval", "reopen", "stuck"):
            outs.append(body)
    return outs


def _run(name, actions):
    doc = yaml.safe_load((PROGRAMS / name).read_text())
    root = _root(doc)
    dec = ScriptedDecoder(_script(actions))
    log = []
    rt = Runtime(lambda lam: ModelAgent(dec, log=log) if lam is root else OracleAgent(oracle, lam))
    out, value = rt.run_root(root)
    return doc, dec, log, out, value


def test_two_phase_decoding_runs_a_program():
    name = "06-map-with-rubric.yaml"
    actions = [a for a, _ in parse_trace(yaml.safe_load((PROGRAMS / name).read_text())["canonical_trace"])]
    doc, dec, log, out, value = _run(name, actions)
    assert not dec.violations, dec.violations
    assert out.kind == "done" and value == doc["expect"]["value"]
    # header and body are separate constrained calls, and the body call continues the header
    set_calls = [c for c in dec.calls if c[2].startswith("fn:")]
    assert set_calls and set_calls[0][0].endswith("set return : Map<Text, Label>\n")
    assert "instructions  Text" in dec.calls[0][0]
    assert "<|" not in dec.calls[0][0]            # the generic wrapper uses no special tokens
    assert "<user>\nRESULT\nok" in dec.calls[2][0]  # results return as user turns


def test_rejected_sample_is_discarded_and_resampled():
    name = "06-map-with-rubric.yaml"
    good = [a for a, _ in parse_trace(yaml.safe_load((PROGRAMS / name).read_text())["canonical_trace"])]
    # a grammar-valid but ill-typed copy: Text into a Text[] slot. Validation rejects it.
    actions = good[:1] + ["copy args/rubric to return/over"] + good[1:]
    doc, dec, log, out, value = _run(name, actions)
    kinds = [(l["action"].splitlines()[0], l["kind"], l["attempt"]) for l in log]
    assert ("copy args/rubric to return/over", "rejected", 0) in kinds
    assert out.kind == "done" and value == doc["expect"]["value"]
    # the rejected action never entered the context shown to the model
    assert all("copy args/rubric to return/over" not in c[0] for c in dec.calls)


def test_stuck_quiesces_with_the_note():
    doc, dec, log, out, value = _run("16-quiesce-with-note.yaml",
                                     ["stuck\nThe rubric has no rule for claims of 50 EUR or more with a receipt."])
    assert out.kind == "quiesced" and "no rule" in out.detail
