"""Memetic optimiser on natlang task success, smallest runnable version (plans/neuralese/MEMETIC_TASK_DECOMPOSITION.md).

An individual is guidance text plus an optional weight adapter, run on train-split task rows through the collector
and scored by the held eval's verifier (`eval.task_packet.classify_case` via `rl.task_rollouts.score`). The held
packet is never read: `authored_root_row` refuses non-train rows. Decisions are natural-language functions run on
an author model (`proposeGuidance`, and `chooseNext` with `--policy nl`); running, verifying and the gradient step are
crisp.

    python -m natlang_neuralese.rl.memetic_tasks --cases CASES --out DIR --server URL --author URL [--author-model ID]
        [--rows 8 --offset 0 --samples 2 --temperature 1.0 --generations 3 --population 3 --shards 4
         --policy crisp|nl --case-seconds 900 --max-model-requests 32 --max-turns 20]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from ..eval.task_packet import read_jsonl
from .task_rollouts import authored_root_row, build_queue, score

ROOT = Path(__file__).resolve().parents[4]
OPERATORS = ("propose", "refine-adapter")
GUIDANCE_WORDS = 120


# Natural-language functions (policy) ---------------------------------------------------------------------------
PROPOSE_GUIDANCE = """You write guidance for a small language model that runs natural-language programs: it reads a call's
inputs, uses tools (eval for exact computation, return_result to finish) and returns a typed result. The guidance is
added to its system prompt for every call of these tasks.

Current guidance:
{guidance}

Task rows it ran and how they ended (outcome class, the checks that failed, the value it returned, errors):
{digest}

Write revised guidance that fixes the failures without naming these tasks or their answers:
1. Find the most common failure class above and what the model did wrong in it.
2. State, as short numbered steps, what the model should do instead.
3. Keep anything from the current guidance that is not contradicted by the failures.
4. Use at most {words} words.
Reply with the guidance inside <guidance>...</guidance>."""

CHOOSE_NEXT = """You steer a search that improves a small model's program state (guidance text and a weight adapter) by task
success on fixed training tasks. Population (id, fitness = success rate, budget stops, guidance length, adapter, parent,
operator that made it):
{population}

Operator history (operator, fitness gain of its child):
{history}

Operators: propose (rewrite the guidance from the parent's failures, keep its adapter); refine-adapter (one
reinforcement step on the parent's own rollouts, keep its guidance; it needs tasks the parent sometimes but not always
solves: {spread} of the parent's tasks have that spread).
1. Pick the parent most worth improving: usually the fittest, unless a close second has an untried operator.
2. Pick the operator that the history says gains most for its cost; try an untried operator once.
3. Do not pick refine-adapter for a parent with no task spread.
Reply with JSON only: {{"parent": "<id>", "operator": "<operator>", "why": "<one sentence>"}}"""


def author_call(url: str, model: str, prompt: str, max_tokens: int = 4096) -> str:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.7, "chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(5):
        try:
            request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"content-type": "application/json"})
            text = json.load(urllib.request.urlopen(request, timeout=900))["choices"][0]["message"].get("content") or ""
            if text.strip():
                return text
        except Exception as error:  # the author server can be restarting; the caller records a failed operator
            if attempt == 4:
                raise
            print(json.dumps({"event": "author_retry", "error": str(error)[:200]}), flush=True)
            time.sleep(30)
    return ""


def propose_guidance(args, parent: dict, digest: str) -> str:
    text = author_call(args.author, args.author_model, PROPOSE_GUIDANCE.format(
        guidance=parent["guidance"] or "(none)", digest=digest, words=GUIDANCE_WORDS))
    match = re.search(r"<guidance>([\s\S]*?)</guidance>", text)
    words = (match.group(1) if match else text).split()
    if not words:
        raise ValueError("author returned no guidance")
    return " ".join(words[:GUIDANCE_WORDS])


# Crisp mechanism -----------------------------------------------------------------------------------------------
def default_system_prompt() -> str:
    script = ("import(process.argv[1]).then(m => process.stdout.write(m.defaultSystemPrompt))")
    return subprocess.run(["node", "-e", script, str(ROOT / "ts-host/dist/teacher/collector.js")], check=True,
                          capture_output=True, text=True).stdout


def guided_system(base: str, guidance: str) -> str:
    """The runtime's program-guidance wrapper (ts-host adaptation/prompts.ts programGuidance)."""
    return base + ("\n\n<natlang_program_guidance>\n" + guidance + "\n</natlang_program_guidance>\n" if guidance else "")


def evaluate(args, ind: dict, packet: Path, base_system: str) -> dict:
    """Run the individual on every packet row (`samples` each, the same seeds for every individual) and verify."""
    out = Path(args.out) / "evals" / ind["id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "packet.jsonl").write_text(packet.read_text())
    system = out / "system.md"
    system.write_text(guided_system(base_system, ind["guidance"]))
    entries = build_queue(out / "packet.jsonl", args.rows, args.samples, out, temperature=args.temperature,
                          case_seconds=args.case_seconds, max_model_requests=args.max_model_requests,
                          max_turns=args.max_turns, seed_base=args.seed_base)
    for entry in entries:
        entry["system_file"] = str(system)
    (out / "queue.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))
    chat = []
    if ind.get("adapter"):
        (out / "chat-config.json").write_text(json.dumps({"x_natlang_adapters": [{"id": ind["adapter"], "scale": 1}]}))
        chat = ["--chat-request-config", str(out / "chat-config.json")]
    shards = out / "shards"
    shards.mkdir(exist_ok=True)
    started = time.time()
    procs = []
    for s in range(args.shards):
        (shards / f"queue-{s}.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries[s::args.shards]))
        log = open(shards / f"queue-{s}.log", "w")
        procs.append(subprocess.Popen([sys.executable, str(ROOT / "scripts/run_bonsai_queue.py"), str(shards / f"queue-{s}.jsonl"),
                                       str(shards / f"journal-{s}.jsonl"), "--runtime", str(ROOT / "ts-host"),
                                       "--server", args.server, "--chat-completions-url", args.server.rstrip("/") + "/v1/chat/completions",
                                       "--model-id", "lfm", "--model-concurrency", "2", *chat], stdout=log, stderr=subprocess.STDOUT))
    for proc in procs:
        proc.wait()
    (out / "journal.jsonl").write_text("".join((shards / f"journal-{s}.jsonl").read_text()
                                               for s in range(args.shards) if (shards / f"journal-{s}.jsonl").exists()))
    report = score(out, out / "rollouts.jsonl")
    rewards = read_jsonl(out / "rewards.jsonl")
    scored = [r for r in rewards if r["reward"] is not None]
    fit = {"fitness": sum(r["reward"] for r in scored) / len(scored) if scored else 0.0,
           "successes": report["successes"], "scored": report["scored"], "rollouts": report["rollouts"],
           "budget_stops": report["outcome_counts"].get("budget_exhaustion", 0),
           "outcomes": report["outcome_counts"], "groups_with_spread": report["groups_with_spread"],
           "per_row": [sum(r["reward"] == 1.0 for r in rewards if r["group"] == g) for g in range(args.rows)],
           "seconds": round(time.time() - started, 1), "dir": str(out)}
    fit["guidance_reached_model"] = guidance_reached(out, ind["guidance"])
    return fit


def guidance_reached(out: Path, guidance: str) -> bool | None:
    """Whether the guidance text is in the recorded model contexts (it must survive prompt rendering)."""
    if not guidance:
        return None
    probe = guidance.split()[:6]
    for path in out.glob("row-*/sample-*/results.jsonl"):
        for row in read_jsonl(path):
            for step in row.get("trajectory") or []:
                if " ".join(probe) in json.dumps(step.get("context") or [], ensure_ascii=False):
                    return True
    return False


def failure_digest(fit: dict, limit: int = 6) -> str:
    """Failed rollouts of an evaluation: outcome class, failed checks, returned value, errors. Never the gold."""
    out, items = Path(fit["dir"]), []
    for reward in read_jsonl(out / "rewards.jsonl"):
        if reward["reward"] != 0.0 or len(items) >= limit:
            continue
        path = out / f"row-{reward['group']:03d}" / f"sample-{reward['sample']:02d}" / "results.jsonl"
        result = (read_jsonl(path) or [{}])[0] if path.exists() else {}
        outcome = result.get("outcome") or {}
        failed = [k for k, v in (outcome.get("checks") or {}).items() if v is False]
        errors = [str(f.get("message", ""))[:200] for f in outcome.get("scope_failures") or [] if isinstance(f, dict)]
        diagnostics = [str(d)[:200] for a in outcome.get("action_ledger") or [] if isinstance(a, dict)
                       for d in (a.get("diagnostics") or [])][-3:]
        items.append(f"- task family {reward['family']}: {reward['outcome']}; failed checks {failed or 'none'}; "
                     f"returned {json.dumps(outcome.get('value'), ensure_ascii=False)[:300]}; errors {errors[:2] + diagnostics}")
    return "\n".join(items) or "(no failures)"


def spread_of(fit: dict) -> int:
    """Rows the individual sometimes but not always solves (the GRPO groups with spread)."""
    return fit["groups_with_spread"]


def choose_next(args, population: list[dict], history: list[dict], rng: random.Random) -> dict:
    """chooseNext: crisp default (tournament of the two best, UCB over operators) or the NL function (--policy nl)."""
    ranked = sorted(population, key=lambda i: (-i["fit"]["fitness"], i["fit"]["budget_stops"]))
    crisp_parent = ranked[0] if len(ranked) == 1 or rng.random() < 0.75 else ranked[1]
    allowed = [op for op in OPERATORS if op != "refine-adapter" or spread_of(crisp_parent["fit"])]
    stats = {op: [h["gain"] for h in history if h["operator"] == op] for op in allowed}
    untried = [op for op in allowed if not stats[op]]
    t = len(history) + 1
    crisp_op = untried[0] if untried else max(allowed, key=lambda op: sum(stats[op]) / len(stats[op]) +
                                                  math.sqrt(2 * math.log(t) / len(stats[op])))
    crisp = {"parent": crisp_parent["id"], "operator": crisp_op, "policy": "crisp"}
    if args.policy != "nl":
        return crisp
    table = "\n".join(f"{i['id']}: fitness {i['fit']['fitness']:.3f}, budget stops {i['fit']['budget_stops']}, "
                      f"guidance {len(i['guidance'].split())} words, adapter {'yes' if i.get('adapter') else 'no'}, "
                      f"parent {i.get('parent')}, operator {i.get('operator')}, spread rows {spread_of(i['fit'])}" for i in population)
    hist = "\n".join(f"{h['operator']}: {h['gain']:+.3f}" for h in history) or "(none yet)"
    text = author_call(args.author, args.author_model, CHOOSE_NEXT.format(population=table, history=hist,
                                                                          spread=spread_of(ranked[0]["fit"])), 2048)
    try:
        answer = json.loads(re.search(r"\{[\s\S]*\}", text).group(0))
        parent = next(i for i in population if i["id"] == answer["parent"])
        if answer["operator"] not in OPERATORS or (answer["operator"] == "refine-adapter" and not spread_of(parent["fit"])):
            raise ValueError(f"operator {answer.get('operator')!r} not allowed for {parent['id']}")
        return {"parent": parent["id"], "operator": answer["operator"], "policy": "nl", "why": str(answer.get("why", ""))[:300]}
    except Exception as error:  # an invalid answer falls back to the crisp choice, recorded
        return {**crisp, "policy": "crisp-fallback", "nl_error": str(error)[:200], "nl_answer": text[:300]}


def refine_adapter(args, parent: dict, child_id: str) -> str:
    """One GRPO round (rl/grpo.py) on the parent's own verified rollouts; returns the new adapter block ID."""
    out = Path(args.out) / "grpo" / child_id
    out.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, "-m", "natlang_neuralese.rl.grpo"]
    adapter = parent.get("adapter")
    if not adapter:
        subprocess.run(base + ["init", "--server", args.server, "--kind", "xs", "--rank", "8", "--out", str(out / "init")],
                       check=True, capture_output=True, text=True)
        adapter = json.loads((out / "init" / "adapter.json").read_text())["adapter"]
    run = subprocess.run(base + ["round", "--server", args.server, "--rollouts", str(Path(parent["fit"]["dir"]) / "rollouts.jsonl"),
                                 "--adapter", adapter, "--out", str(out / "round"), "--lr", str(args.adapter_lr),
                                 "--beta", "0.02", "--steps", "1"], capture_output=True, text=True)
    (out / "round.log").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(f"grpo round failed: {(run.stderr or run.stdout)[-300:]}")
    return json.loads((out / "round" / "adapter.json").read_text())["adapter"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--author-model", default="author")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=3)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--policy", choices=("crisp", "nl"), default="nl")
    parser.add_argument("--case-seconds", type=int, default=900)
    parser.add_argument("--max-model-requests", type=int, default=32)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=13000)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = [authored_root_row(row) for row in read_jsonl(args.cases)[args.offset:args.offset + args.rows]]
    packet = out / "packet.jsonl"
    packet.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    (out / "run.json").write_text(json.dumps({"schema": "natlang.memetic-tasks/1", **{k: str(v) for k, v in vars(args).items()},
                                              "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
                                              "row_ids": [r["id"] for r in rows]}, indent=1) + "\n")
    base_system = default_system_prompt()
    rng = random.Random(args.seed_base)
    population: list[dict] = []
    history: list[dict] = []
    seed = {"id": "ind-0", "guidance": "", "adapter": None, "parent": None, "operator": "seed"}
    seed["fit"] = evaluate(args, seed, packet, base_system)
    population.append(seed)
    print(json.dumps({"event": "seed", "fit": {k: v for k, v in seed["fit"].items() if k != "dir"}}), flush=True)
    serial = 0
    for generation in range(args.generations):
        choice = choose_next(args, population, history, rng)
        parent = next(i for i in population if i["id"] == choice["parent"])
        serial += 1
        child = {"id": f"ind-{serial}", "guidance": parent["guidance"], "adapter": parent.get("adapter"),
                 "parent": parent["id"], "operator": choice["operator"]}
        started = time.time()
        try:
            if choice["operator"] == "propose":
                child["guidance"] = propose_guidance(args, parent, failure_digest(parent["fit"]))
            else:
                child["adapter"] = refine_adapter(args, parent, child["id"])
            child["fit"] = evaluate(args, child, packet, base_system)
        except Exception as error:
            history.append({"operator": choice["operator"], "gain": 0.0})
            step = {"generation": generation, "choice": choice, "error": str(error)[:500]}
            print(json.dumps(step), flush=True)
            with open(out / "steps.jsonl", "a") as stream:
                stream.write(json.dumps(step) + "\n")
            continue
        gain = child["fit"]["fitness"] - parent["fit"]["fitness"]
        history.append({"operator": choice["operator"], "gain": gain})
        population = sorted(population + [child], key=lambda i: (-i["fit"]["fitness"], i["fit"]["budget_stops"]))[:args.population]
        step = {"schema": "natlang.memetic-task-step/1", "generation": generation, "choice": choice,
                "before": {"id": parent["id"], "fitness": parent["fit"]["fitness"]},
                "after": {"id": child["id"], "fitness": child["fit"]["fitness"], "guidance": child["guidance"], "adapter": child["adapter"],
                          "per_row": child["fit"]["per_row"], "outcomes": child["fit"]["outcomes"],
                          "guidance_reached_model": child["fit"]["guidance_reached_model"]},
                "gain": gain, "kept": child in population, "seconds": round(time.time() - started, 1)}
        print(json.dumps(step), flush=True)
        with open(out / "steps.jsonl", "a") as stream:
            stream.write(json.dumps(step) + "\n")
        (out / "population.json").write_text(json.dumps(population, indent=1) + "\n")
    best = population[0]
    summary = {"best": {k: best[k] for k in ("id", "guidance", "adapter", "parent", "operator")},
               "best_fit": {k: v for k, v in best["fit"].items() if k != "dir"},
               "seed_fit": {k: v for k, v in seed["fit"].items() if k != "dir"}, "history": history}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
