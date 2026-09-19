"""The semantic highlighter code base (codebases/highlighter): natlang reading natlang source."""
import re
from pathlib import Path

from natlang import gbnf
from natlang.gen.policy import native_text
from natlang.host import load
from natlang.native import call_grammar
from natlang.runtime import Runtime
from natlang.surface import ToolSurface
from natlang.values import dump

ROOT = Path(__file__).resolve().parent.parent
S = ToolSurface()


def line_role(a):                      # stands in for the model's judgment
    l, fns = a["line"].strip(), a["functions"]
    called = any(re.search(rf"\b{re.escape(f)}\(", l) for f in fns)
    if not l:
        return "blank"
    if l.startswith("#"):
        return "comment"
    if l.startswith("function "):
        return "signature"
    if l.startswith("return"):
        return "return"
    if re.match(r"(if|else|otherwise)\b", l):
        return "condition"
    if l.startswith("repeat") or "until" in l.split("#")[0]:
        return "repeat"
    if "for each" in l and called:
        return "call_each"
    if called:
        return "call"
    if "# exact" in l:
        return "exact"
    return "prose_step" if "=" in l else "leaf_text"


class Interpreter:
    def __init__(self, lam):
        self.lam = lam

    def do(self, session, name, args):
        assert gbnf.accepts(call_grammar(S.tools(session)), native_text([(name, args)])), (self.lam.fn_name, name, args)
        r = S.apply(session, name, args)
        assert r.kind not in ("rejected", "refused", "error", "quiesced"), (self.lam.fn_name, name, args, r.text)
        return r

    def run(self, session):
        f, d = self.lam.fn_name, lambda n, a: self.do(session, n, a)
        if f == "line_role":
            d("write", {"path": "return", "type": "Role", "value": line_role(self.lam.in_)})
        elif f == "highlight":
            d("call", {"function": "highlight_file", "to": "return", "over": "args/files"})
        else:
            d("call", {"function": "split_source", "to": "let/parts", "inputs": {"file": "args/file"}})
            if d("read", {"path": "let/parts/is_code"}).value:
                d("write", {"path": "let/roles", "type": "Role[]", "value": []})
            else:
                d("call", {"function": "line_role", "to": "let/roles", "over": "let/parts/lines",
                           "inputs": {"functions": "let/parts/functions"}})
            d("call", {"function": "render_html", "to": "return/html",
                       "inputs": {"file": "args/file", "parts": "let/parts", "roles": "let/roles"}})
            d("write", {"path": "return/roles", "type": "Role[]", "source": "let/roles"})
            d("write", {"path": "return/path", "type": "Text", "source": "args/file/path"})
        assert session.finish()


def test_highlighting_the_triage_example():
    base = ROOT / "examples" / "triage"
    paths = [base / "main.nl", base / "main" / "summarize.nl", base / "main" / "classify.nl", ROOT / "examples" / "std" / "count_true.ts"]
    files = [{"path": str(p.relative_to(ROOT)), "text": p.read_text()} for p in paths]
    rt = Runtime(lambda lam: Interpreter(lam), max_episodes=400)
    out, value = rt.run_root(load(ROOT / "codebases" / "highlighter" / "highlight.nl", {"files": files}))
    assert out.kind == "done", out.detail
    result = dump(value)
    main = result[0]
    assert main["roles"][0] == "signature" and "call_each" in main["roles"] and "condition" in main["roles"]
    assert '<span class="nl-fn">classify</span>' in main["html"] and "nl-role-call_each" in main["html"]
    assert result[2]["roles"] and set(result[2]["roles"]) == {"leaf_text"}          # a leaf is prose, not steps
    assert result[3]["roles"] == [] and "nl-role-code" in result[3]["html"]          # a .ts file is not judged
    out_dir = ROOT / "runs" / "highlight"
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in result:
        (out_dir / (r["path"].replace("/", "__") + ".html")).write_text(r["html"])
