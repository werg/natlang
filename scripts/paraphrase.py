#!/usr/bin/env python3
"""Teacher-controlled surface diversity, with a round-trip check.

For every hand-written instruction text of every family, ask the teacher for paraphrases, then keep a
paraphrase only if the TEACHER ITSELF, executing the paraphrased program through the real harness, reaches
the known answer on fresh instances. Accepted paraphrases go to data/paraphrases.json, which the
generator draws from. Usage: scripts/paraphrase.py --server http://127.0.0.1:8081 --k 4 --trials 2
"""
import argparse, json, random, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.decoder import LlamaServerDecoder
from natlang.gen import programs as P
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.types import TypeEnv
from natlang.values import coerce, dump, load_program

ASK = """Rewrite the following task instruction in {k} different ways, as different people would phrase it
(a terse engineer, a polite office worker, a checklist writer, someone writing quickly). Keep the meaning
EXACTLY the same. Keep every `backticked/path` exactly as it is, character for character. Do not add or
remove requirements. Answer with a JSON list of {k} strings and nothing else.

Instruction:
{text}"""


def ask_paraphrases(dec, text, k):
    turn = dec.chat([{"role": "user", "content": ASK.format(k=k, text=text)}], [], temperature=0.9, max_tokens=900)
    m = re.search(r"\[.*\]", turn.text, re.S)
    try:
        out = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        out = []
    ticks = sorted(re.findall(r"`[^`]+`", text))
    return [o.strip() for o in out if isinstance(o, str) and sorted(re.findall(r"`[^`]+`", o)) == ticks
            and o.strip() != text.strip()]


def teacher_solves(dec, family, base, para, rng, log):
    fn = P.FAMILIES[family]
    prog = fn(rng, text={"base": base, "text": para}) if family == "crisp_scalar" else fn(rng, text=para)
    root = load_program(prog.root)
    env = root.env(TypeEnv())
    for name, value in prog.inputs.items():
        root.in_[name] = coerce(value, root.type.params.get(name)[0], env, yaml=False, path=f"args/{name}")
    rt = Runtime(lambda lam: ToolAgent(dec, temperature=0.6, log=log), max_episodes=24)
    out, value = rt.run_root(root)
    got = dump(value) if out.kind == "done" else None
    ok = got == prog.expected or (isinstance(got, float) and abs(got - prog.expected) < 0.011)
    return ok, out.kind, got, prog.expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--thinking", type=int, default=512)
    ap.add_argument("--families", nargs="*", default=list(P.FAMILIES))
    ap.add_argument("--out", type=Path, default=Path("data/paraphrases.json"))
    a = ap.parse_args()
    dec = LlamaServerDecoder(a.server, timeout=900,
                             chat_extra={"thinking_budget_tokens": a.thinking, "top_p": 0.95, "top_k": 20})
    rng = random.Random(11)
    for fam in a.families:                      # populate BASE_TEXTS
        P.FAMILIES[fam](random.Random(0))
    accepted = json.loads(a.out.read_text()) if a.out.exists() else {}
    report = []
    for fam in a.families:
        for base in P.BASE_TEXTS[fam]:
            if len(accepted.get(base, [])) >= a.k - 1:      # resumable: this base is already covered
                continue
            t0 = time.time()
            try:
                cands = ask_paraphrases(dec, base, a.k)
            except OSError as e:                            # server restarting: wait and move on
                print(f"[{fam}] server error {e!r}; waiting", flush=True); time.sleep(90); continue
            for para in cands:
                if para in accepted.get(base, []):
                    continue
                log, results = [], []
                for _ in range(a.trials):
                    try:
                        results.append(teacher_solves(dec, fam, base, para, rng, log))
                    except OSError as e:
                        print(f"[{fam}] server error {e!r}; waiting", flush=True); time.sleep(90)
                        results.append((False, "server-error", None, None))
                    if not results[-1][0]:
                        break
                keep = all(r[0] for r in results) and len(results) == a.trials
                report.append({"family": fam, "base": base, "paraphrase": para, "kept": keep,
                               "results": [r[1:] for r in results]})
                if keep:
                    accepted.setdefault(base, []).append(para)
                    a.out.parent.mkdir(parents=True, exist_ok=True)
                    a.out.write_text(json.dumps(accepted, indent=1, ensure_ascii=False))
                print(f"[{fam}] {'KEEP' if keep else 'drop'}  {para[:90]!r}  {[r[1] for r in results]}", flush=True)
            print(f"[{fam}] base done in {time.time()-t0:.0f}s: {base[:60]!r}", flush=True)
    Path("data/paraphrase_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    kept = sum(1 for r in report if r["kept"])
    print(f"kept {kept} of {len(report)} candidate paraphrases; usage: {dec.usage}")


if __name__ == "__main__":
    main()
