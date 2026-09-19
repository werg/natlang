"""The web server code base (codebases/webserver): a fold over requests; routing, sessions, forms, errors."""
from pathlib import Path

import yaml

from natlang import gbnf
from natlang.gen.policy import native_text
from natlang.host import load_fold
from natlang.native import call_grammar
from natlang.runtime import Runtime
from natlang.surface import ToolSurface
from natlang.values import dump

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "codebases" / "webserver"
S = ToolSurface()


def leaf(fn, a):                                   # stands in for the model
    if fn == "page_content":
        items = "".join(f"<li>{e['author']}: {e['message']}</li>" for e in a["entries"])
        return f"<h2>{a['purpose'].split(':')[0]}</h2><p>{a['about']}</p><ul>{items}</ul>" + ("<p>Welcome back!</p>" if a["session"]["visits"] > 1 else "")
    if fn == "review_submission":
        msg = a["form"].get("message", "").strip()
        bad = not msg or "<script" in msg.lower() or "buy now" in msg.lower()
        return {"accept": not bad, "reason": "Thanks!" if not bad else "That does not look like a guestbook message.",
                "author": a["form"].get("author", "").strip() or "anonymous", "message": msg}
    return f"<p>{a['review']['reason']}</p><p><a href=\"/\">home</a></p>"


class Interpreter:
    """Follows handle.nl as a good interpreter would: exact steps by calls, one branch carried out."""
    def __init__(self, lam, log):
        self.lam, self.log = lam, log

    def do(self, session, name, args):
        assert gbnf.accepts(call_grammar(S.tools(session)), native_text([(name, args)])), (self.lam.fn_name, name, args)
        r = S.apply(session, name, args)
        assert r.kind not in ("rejected", "refused", "error", "quiesced"), (self.lam.fn_name, name, args, r.text)
        self.log.append(args.get("function"))
        return r

    def run(self, session):
        f, d = self.lam.fn_name, lambda n, a: self.do(session, n, a)
        call = lambda fn, to, **inputs: d("call", {"function": fn, "to": to, "inputs": inputs})
        if f != "handle":
            from natlang.types import format_type
            d("write", {"path": "return", "type": format_type(self.lam.type.returns), "value": leaf(f, self.lam.in_)})
            assert session.finish()
            return
        call("parse_request", "let/req", item="args/item")
        call("session_for", "let/session", acc="args/acc", req="let/req")
        call("match_route", "let/route", routes="args/acc/routes", req="let/req")
        kind = d("read", {"path": "let/route/kind"}).value
        site = "args/acc"
        if kind == "static":
            call("static_response", "let/response", acc="args/acc", route="let/route")
        elif kind == "page":
            call("page_content", "let/content", purpose="let/route/purpose", site_name="args/acc/name", about="args/acc/about",
                 entries="args/acc/entries", session="let/session")
            call("wrap_page", "let/response", acc="args/acc", route="let/route", content="let/content", session="let/session")
        elif kind == "form":
            call("review_submission", "let/review", purpose="let/route/purpose", form="let/req/form")
            call("apply_submission", "let/site", acc="args/acc", review="let/review")
            site = "let/site"
            call("submission_page", "let/content", purpose="let/route/purpose", review="let/review")
            call("wrap_page", "let/response", acc="let/site", route="let/route", content="let/content", session="let/session")
        else:
            call("error_response", "let/response", route="let/route")
        call("respond", "let/sent", id="args/item/id", response="let/response")
        call("log_request", "return", site=site, req="let/req", session="let/session", response="let/response")
        assert session.finish()


def req(i, method, path, body="", cookie=""):
    return {"id": f"r{i}", "method": method, "path": path, "headers": {"Cookie": cookie} if cookie else {}, "body": body}


def test_a_session_of_requests():
    sent, log = {}, []
    events = [req(1, "GET", "/"), req(2, "GET", "/style.css"), req(3, "GET", "/nope"), req(4, "GET", "/submit"),
              req(5, "POST", "/submit", "author=Ada&message=Hello+from+Ada"),
              req(6, "POST", "/submit", "author=x&message=%3Cscript%3Ealert(1)%3C%2Fscript%3E BUY NOW")]

    def source():
        for e in events:
            yield e
        cookie = sent["r1"]["headers"]["Set-Cookie"].split(";")[0]          # the visitor comes back with the cookie
        yield req(7, "GET", "/", cookie=cookie)

    rt = Runtime(lambda lam: Interpreter(lam, log), max_episodes=400,
                 capabilities={"http.respond": lambda a: sent.__setitem__(a[0], a[1])})
    site = yaml.safe_load((WEB / "site.yaml").read_text())
    out, value = rt.run_root(load_fold(WEB / "handle.nl", site, source()))
    assert out.kind == "done", out.detail
    final = dump(value)
    assert [sent[f"r{i}"]["status"] for i in range(1, 8)] == [200, 200, 404, 405, 200, 200, 200]
    assert sent["r4"]["headers"]["Allow"] == "POST" and sent["r2"]["headers"]["Content-Type"].startswith("text/css")
    assert final["entries"] == [{"author": "Ada", "message": "Hello from Ada"}]          # the second submission was refused
    assert "Ada: Hello from Ada" in sent["r7"]["body"] and "Welcome back!" in sent["r7"]["body"]
    assert "Welcome back!" not in sent["r1"]["body"] and len(final["log"]) == 7
    assert "page_content" not in log[log.index("static_response") - 3: log.index("static_response")]   # one branch only
