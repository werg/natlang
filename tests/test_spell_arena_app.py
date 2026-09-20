"""Semantic spell proposals pass through the natlang call tree and atomic exact resolver."""
from pathlib import Path

from natlang.host import load
from natlang.runtime import Runtime
from natlang.surface import ToolSurface
from natlang.types import format_type
from natlang.values import dump


ENTRY = Path(__file__).resolve().parent.parent / "codebases/spell_arena/cast.nl"
SURFACE = ToolSurface()
WORLD = {"revision": 4, "energy": 6, "actors": [
    {"id": "caster", "hp": 10, "shield": 0, "x": 0, "y": 0},
    {"id": "ogre", "hp": 8, "shield": 1, "x": 2, "y": 0},
]}
RULES = {"max_damage": 4, "max_shield": 3, "max_move": 3,
         "damage_cost": 1, "shield_cost": 1, "move_cost": 1}


class Interpreter:
    def __init__(self, lam, meaning):
        self.lam, self.meaning = lam, meaning

    def run(self, session):
        def do(name, args):
            result = SURFACE.apply(session, name, args)
            assert result.kind not in ("rejected", "refused", "error", "quiesced"), (name, args, result.text)
            return result

        if self.lam.fn_name == "interpret":
            do("write", {"path": "return", "type": format_type(self.lam.type.returns), "value": self.meaning})
        else:
            do("call", {"function": "interpret", "to": "let/meaning",
                        "inputs": {"utterance": "args/utterance", "observation": "args/observation",
                                   "rules": "args/rules"}})
            do("call", {"function": "resolve", "to": "return",
                        "inputs": {"observation": "args/observation", "rules": "args/rules",
                                   "meaning": "let/meaning"}})
        assert session.finish()


def cast(meaning, utterance="Strike the ogre and shield me"):
    rt = Runtime(lambda lam: Interpreter(lam, meaning))
    outcome, value = rt.run_root(load(ENTRY, {"utterance": utterance, "observation": WORLD, "rules": RULES}))
    assert outcome.kind == "done", outcome.detail
    return dump(value)


def effect(kind, target, amount=0, x=0, y=0):
    return {"kind": kind, "target": target, "amount": amount, "x": x, "y": y}


def plan(effects, revision=4):
    return {"status": "plan", "plan": {"revision": revision, "effects": effects},
            "explanation": "An ordered spell proposal."}


def test_composed_cast_applies_once_and_conserves_energy():
    result = cast(plan([effect("damage", "ogre", 3), effect("shield", "caster", 2)]))
    assert result["status"] == "applied" and result["cost"] == 5
    assert result["world"]["revision"] == 5 and result["world"]["energy"] == 1
    assert result["world"]["actors"][1]["hp"] == 6
    assert result["world"]["actors"][1]["shield"] == 0
    assert result["world"]["actors"][0]["shield"] == 2


def test_invalid_later_effect_and_unaffordable_plan_have_no_partial_commit():
    invalid = cast(plan([effect("damage", "ogre", 2), effect("move", "caster", 0, 2, 0)]))
    assert invalid["status"] == "invalid" and invalid["world"] == WORLD and invalid["cost"] == 0
    expensive = cast(plan([effect("damage", "ogre", 4), effect("shield", "caster", 3)]))
    assert expensive["status"] == "insufficient" and expensive["world"] == WORLD


def test_stale_and_ambiguous_spells_leave_world_unchanged():
    stale = cast(plan([effect("damage", "ogre", 2)], revision=3))
    assert stale["status"] == "stale" and stale["world"] == WORLD
    clarify = cast({"status": "clarify", "plan": {"revision": 4, "effects": []},
                    "explanation": "Which actor should receive the shield?"}, "Shield them")
    assert clarify["status"] == "clarify" and clarify["world"] == WORLD
