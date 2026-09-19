import random

from natlang.checks import grade
from natlang.gen import programs as P
from scripts.generate import run_program


def test_crisp_checks_and_status():
    crisp = {"checks": [{"kind": "crisp", "code": "wordCount(value) <= 4"}]}
    assert grade(crisp, "done", "Shop closed on Monday")[0] == "yes"
    assert grade(crisp, "done", "The shop will be closed on Monday")[0] == "no"
    assert grade(crisp, "quiesced", None)[0] == "no"
    assert grade({"status": "quiesced"}, "done", "reject")[0] == "no"
    assert grade({"status": "quiesced"}, "quiesced", None)[0] == "yes"


def test_judge_checks_need_a_judge():
    exp = {"checks": [{"kind": "judge", "question": "Is it about Monday?", "answer": True}]}
    assert grade(exp, "done", "x")[0] == "?"
    assert grade(exp, "done", "x", judge=lambda s, q: True)[0] == "yes"
    assert grade(exp, "done", "x", judge=lambda s, q: False)[0] == "no"


def test_undetermined_instances_end_in_a_blocker():
    rng, seen = random.Random(2), 0
    for _ in range(80):
        prog = P.extract(rng)
        samples, _ = run_program(prog)                     # asserts quiesced + grammar-valid turns
        if prog.expected == P.BLOCKED:
            seen += 1
            assert samples[-1]["skill"] == "report_blocker"
    assert seen > 0
