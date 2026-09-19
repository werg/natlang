"""The legal-move code base (codebases/legal_move): exact rules in code, fuzzy reading by leaves, an early return,
and a blocker from a leaf that travels up instead of being guessed around."""
from pathlib import Path

from natlang import gbnf
from natlang.gen.policy import native_text
from natlang.host import load
from natlang.native import call_grammar
from natlang.runtime import Runtime
from natlang.surface import ToolSurface
from natlang.types import format_type
from natlang.values import dump

ROOT = Path(__file__).resolve().parent.parent
S = ToolSurface()
E_ = "empty"
POSITIONS = {"X holds the centre and the top-left corner. O has the top-right corner and the bottom-left corner.":
                 ["X", E_, "O", E_, "X", E_, "O", E_, E_],
             "X has the whole top row. O has the two left cells of the middle row.": ["X", "X", "X", "O", "O", E_, E_, E_, E_]}
MOVES = {"bottom right": 9, "the centre": 5, "somewhere on the left": None}


class Interpreter:
    def __init__(self, lam):
        self.lam = lam

    def do(self, session, name, args):
        assert gbnf.accepts(call_grammar(S.tools(session)), native_text([(name, args)])), (self.lam.fn_name, name, args)
        r = S.apply(session, name, args)
        assert r.kind not in ("rejected", "refused", "error"), (self.lam.fn_name, name, args, r.text)
        return r

    def run(self, session):
        f, a, d = self.lam.fn_name, self.lam.in_, lambda n, x: self.do(session, n, x)
        if f in ("read_position", "read_move"):
            v = POSITIONS[a["position"]] if f == "read_position" else MOVES[a["move"]]
            if v is None:
                return d("report_blocker", {"missing": "The move does not name exactly one cell: 'left' could be three cells."}).text
            d("write", {"path": "return", "type": format_type(self.lam.type.returns), "value": v})
            assert session.finish()
            return None
        d("call", {"function": "read_position", "to": "let/cells", "inputs": {"position": "args/position"}})
        d("call", {"function": "board_problem", "to": "let/problem", "inputs": {"cells": "let/cells", "player": "args/player"}})
        if d("read", {"path": "let/problem"}).value != "":
            d("call", {"function": "draw", "to": "return/board", "inputs": {"cells": "let/cells"}})
            d("write", {"path": "return/reason", "type": "Text", "source": "let/problem"})
            d("write", {"path": "return/legal", "type": "Bool", "value": False})
            d("write", {"path": "return/wins", "type": "Bool", "value": False})
        else:
            r = d("call", {"function": "read_move", "to": "let/target", "inputs": {"move": "args/move"}})
            if r.kind == "quiesced":                       # the callee reported a blocker: pass it up, do not guess
                return d("report_blocker", {"missing": "read_move could not tell which cell is meant: " + r.text[:160]}).text
            d("call", {"function": "judge_move", "to": "return", "inputs": {"cells": "let/cells", "target": "let/target", "player": "args/player"}})
        assert session.finish()
        return None


def check(position, move, player):
    rt = Runtime(lambda lam: Interpreter(lam))
    out, value = rt.run_root(load(ROOT / "codebases" / "legal_move" / "check_move.nl",
                                  {"position": position, "move": move, "player": player}))
    return out, (dump(value) if out.kind == "done" else None)


def test_a_winning_move_an_occupied_cell_and_a_finished_game():
    p1, p2 = list(POSITIONS)
    out, v = check(p1, "bottom right", "X")
    assert v["legal"] and v["wins"] and v["board"].splitlines()[2] == "O . X"
    assert check(p1, "the centre", "X")[1]["reason"] == "cell 5 is already taken by X"
    assert check(p1, "bottom right", "O")[1]["reason"] == "it is X's turn, not O's"
    assert check(p2, "bottom right", "O")[1] == {"legal": False, "reason": "the game is already over", "wins": False,
                                                  "board": "X X X\nO O .\n. . ."}


def test_an_unclear_move_is_a_blocker_not_a_guess():
    out, v = check(list(POSITIONS)[0], "somewhere on the left", "X")
    assert out.kind == "quiesced" and "which cell" in out.detail
