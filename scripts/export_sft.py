#!/usr/bin/env python3
"""Turn the reference corpus into supervised pairs of text, rendered by the model's own chat template exactly as at
inference (the llama.cpp server's /apply-template with the official template and the turn's tools).

  scripts/export_sft.py data/external_pilot/direct-core-v2-traces.jsonl.gz data/direct-core-v2.sft.jsonl --server http://127.0.0.1:8080 [--limit N] [--every K]
Each output line: {"id", "family", "skill", "prompt", "completion"}; train on the completion only.
"""
import argparse, hashlib, json, sys, time, urllib.error, urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang import gbnf
from natlang.decoder import _cache_stable_tools
from natlang.native import CALL_CLOSE, CALL_OPEN, _strip_private, call_grammar, parse_calls
from natlang.corpus import program_id
from natlang.gen.policy import native_text
from natlang.terminal import reply_only_sample

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--limit", type=int, default=10**9)
    ap.add_argument("--every", type=int, default=1, help="keep every K-th sample")
    ap.add_argument("--terminal-every", type=int, default=1,
                    help="keep every K-th empty terminal reply while retaining every action target")
    ap.add_argument("--workers", type=int, default=1, help="bounded parallel template requests")
    ap.add_argument("--resume", action="store_true", help="append after verifying the last existing ID")
    ap.add_argument("--include-template", action="store_true",
                    help="include programs with provisional generated-text gold (excluded by default)")
    ap.add_argument("--end-token", default="<|im_end|>",
                    help="assistant message terminator used by the selected model template")
    ap.add_argument("--template-id", default="qwen-chatml",
                    help="human-readable identity of the server's chat template")
    ap.add_argument("--cache-stable-tools", action="store_true",
                    help="render the same cache-stable schemas used by live inference")
    ap.add_argument("--synthetic-reasoning", choices=("none", "action"), default="none",
                    help="add a concise verified rationale when a synthetic row has no teacher reasoning")
    ap.add_argument("--require-native-roundtrip", action="store_true",
                    help="reject targets that do not round-trip through the live grammar and parser")
    ap.add_argument("--drop-invalid-actions", action="store_true",
                    help="for legacy migration, record and omit invalid or post-completion action targets")
    a = ap.parse_args()
    if a.workers < 1 or a.every < 1 or a.terminal_every < 1 or a.limit < 1:
        ap.error("workers, every, terminal-every and limit must be positive")
    if a.drop_invalid_actions and not a.require_native_roundtrip:
        ap.error("--drop-invalid-actions requires --require-native-roundtrip")
    if a.drop_invalid_actions and a.resume:
        ap.error("legacy filtering cannot be combined with --resume")

    try:
        props = json.loads(urllib.request.urlopen(a.server + "/props", timeout=10).read())
    except (urllib.error.URLError, ValueError):
        props = {}
    template_source = props.get("chat_template_tool_use") or props.get("chat_template")
    template_hash = hashlib.sha256(template_source.encode()).hexdigest() if isinstance(template_source, str) else None
    renderer = {"version": "llama.cpp-apply-template/2", "template_id": a.template_id,
                "template_sha256": template_hash, "end_token": a.end_token,
                "server": a.server, "include_template": a.include_template,
                "cache_stable_tools": a.cache_stable_tools,
                "terminal_tool_policy": "empty-success-turn-v2",
                "teacher_reasoning_policy": ("render-if-present+synthetic-action-v3"
                                             if a.synthetic_reasoning == "action" else "render-if-present-v1"),
                "native_roundtrip": a.require_native_roundtrip,
                "invalid_action_policy": ("drop-and-audit-v1" if a.drop_invalid_actions else "reject-v1"),
                "native_target_policy": ("splice-or-derive-verified-native-v2" if a.require_native_roundtrip
                                         else "template-serialized-v1")}
    manifest_path = a.dst.with_suffix(a.dst.suffix + ".manifest.json")

    def render(messages, tools):
        shown = _strip_private(tools)
        if a.cache_stable_tools:
            shown = _cache_stable_tools(shown)
        req = urllib.request.Request(a.server + "/apply-template", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"messages": messages, "tools": shown}).encode())
        for attempt in range(5):
            try:
                return json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt"]
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                if attempt == 4:
                    raise
                time.sleep(min(0.25 * 2 ** attempt, 4))

    if renderer["template_sha256"] is None:
        probe = render([{"role": "system", "content": "template identity probe"},
                        {"role": "user", "content": "probe"}], [])
        renderer["template_sha256"] = hashlib.sha256(probe.encode()).hexdigest()
    if a.resume and a.dst.exists():
        prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        same_selection = (prior.get("every", 1) == a.every and
                          prior.get("terminal_every", 1) == a.terminal_every and
                          prior.get("limit", 10**9) == a.limit)
        if prior.get("renderer") != renderer or not same_selection:
            ap.error("resume renderer differs from existing SFT manifest")

    import gzip

    def lines():                                   # a .jsonl file, a .jsonl.gz file, or a directory of shards
        files = sorted(a.src.glob("part-*.jsonl.gz")) if a.src.is_dir() else [a.src]
        for path in files:
            with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
                yield from f

    def parsed_target(s):
        calls = s["target"].get("tool_calls") or []
        expected = []
        for call in calls:
            fn = call["function"]
            args = json.loads(fn.get("arguments") or "{}")
            if not isinstance(args, dict):
                raise ValueError(f"{s['id']}: tool arguments are not an object")
            expected.append((fn["name"], args))
        native = s.get("native_target", "" if not calls else None)
        if native is None:
            # Older teacher projections retained the structured choices and
            # reasoning but predated native text in trajectory IR. Rebuild the
            # canonical syntax, then subject it to the same parser and live
            # grammar checks as newly materialized targets.
            native = native_text(expected) if calls else ""
            s["native_target"] = native
        if calls:
            try:
                actual = parse_calls(native)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"{s['id']}: native target does not parse: {exc}") from exc
            if actual != expected:
                raise ValueError(f"{s['id']}: native target differs from structured target")
            verified = s.get("native_target_grammar_verified") is True and isinstance(
                s.get("native_grammar_sha256"), str)
            if not verified and not gbnf.accepts(call_grammar(s["tools"]), native):
                raise ValueError(f"{s['id']}: live grammar refuses native target")
        elif native.strip():
            raise ValueError(f"{s['id']}: terminal target has nonempty native text")
        return expected

    def action_reasoning(s, calls):
        if s.get("skill") == "checkpoint":
            return ("The conversation has reached a safe continuation boundary while work remains. I should leave "
                    "a short note containing only non-obvious unresolved context; the fresh conversation will receive "
                    "the program, typed workspace, line marks, and effect journal again.")
        if not calls:
            return ("The return slot is filled with a value of the declared type, and the instruction listing "
                    "has no substantive open lines. No further tool action is needed, so I should end the turn.")
        parts = []
        previous = next((m.get("content", "") for m in reversed(s["messages"])
                         if m.get("role") == "tool"), "")
        if previous:
            if any(word in previous.lower() for word in ("reject", "error", "invalid", "does not fit")):
                parts.append("The preceding proposal failed validation, so I must correct the concrete cause rather than repeat or disguise it.")
            else:
                parts.append("The preceding tool result is now part of the typed state, so the next action can use that verified value.")
        else:
            parts.append("I will follow the current instruction and preserve the declared input and return types.")
        for name, args in calls:
            if name in ("call", "run_function", "for_each", "fold", "repeat"):
                shape = " over the selected items" if name in ("for_each", "fold") or "over" in args else ""
                destination = args.get("save_as", args.get("to", "the destination"))
                parts.append(f"The required subproblem is implemented by {args.get('function', 'the named function')}{shape}; its typed result belongs in {destination} so later steps can depend on it.")
            elif name in ("write", "write_value", "copy_value"):
                source = (f" from {args['source']}" if "source" in args else "")
                destination = args.get("destination", args.get("path", "the destination"))
                parts.append(f"The value is fully determined{source}; write it to {destination} as {args.get('type', 'the declared type')} without inventing or coercing a different result.")
            elif name == "run_code":
                parts.append("This is a deterministic algorithm over the supplied values. Execute the stated transformation exactly so sorting, grouping, boundaries, and arithmetic are handled by code rather than mental approximation.")
            elif name == "read":
                parts.append(f"The current listing only previews {args.get('path', 'the required value')}; read it before deciding so no hidden element is guessed or omitted.")
            elif name in ("report_error", "report_blocker"):
                parts.append("The requested result is not derivable from the available instructions and values. Report the precise failure instead of fabricating a value that merely satisfies the schema.")
            elif name in ("mark_done", "mark_lines"):
                parts.append("Close only the instruction lines whose effects are already reflected in state; skipped or unresolved work must remain distinguishable.")
            else:
                parts.append(f"Apply the verified next action: {name}.")
        parts.append("After the tool result returns, I will reassess the updated state before choosing another action.")
        return " ".join(parts)

    def convert(s):
        s = reply_only_sample(s)
        try:
            checked_calls = parsed_target(s) if a.require_native_roundtrip else None
        except ValueError as exc:
            if a.drop_invalid_actions:
                return None, "grammar_or_parser: " + str(exc)
            raise
        prompt = render(s["messages"], s["tools"])
        calls = s["target"].get("tool_calls")
        tool_names = {t["function"]["name"] for t in s["tools"]}
        return_already_written = any(
            m.get("role") == "tool" and "return: written" in (m.get("content") or "")
            for m in s["messages"])
        if calls and return_already_written and not ({"mark_done", "mark_lines"} & tool_names):
            reason = f"post_completion_action: {s['id']} proposed another action after the return was complete"
            if a.drop_invalid_actions:
                return None, reason
            raise ValueError(reason)
        reasoning = s.get("teacher_reasoning")
        if not reasoning and a.synthetic_reasoning == "action":
            reasoning = action_reasoning(s, checked_calls if checked_calls is not None else parsed_target(s))
        if calls or reasoning:        # let the model template serialize both decisions and thinking
            tgt = {**s["target"]}
            if calls:
                tgt["tool_calls"] = [{**c, "id": f"x{j}"} for j, c in enumerate(calls)]
            if reasoning:
                tgt["reasoning_content"] = reasoning
            after = ([{"role": "tool", "tool_call_id": f"x{j}", "content": "X"}
                      for j in range(len(calls))] if calls else [{"role": "user", "content": "X"}])
            full = render(s["messages"] + [tgt] + after, s["tools"])
            if not full.startswith(prompt):
                raise ValueError("template does not preserve the assistant prefix; choose a compatible renderer")
            suffix = full[len(prompt):]
            if calls and a.require_native_roundtrip:
                start, end = suffix.find(CALL_OPEN), suffix.find(CALL_CLOSE)
                if start < 0 or end < start:
                    raise ValueError(f"{s['id']}: rendered completion has no native call wrapper")
                # The LFM Jinja serializer does not escape every Python string
                # correctly (notably code containing single quotes). The IR
                # target has already passed the exact live grammar and parser,
                # so retain the template's reasoning/wrapper placement but use
                # the verified action bytes themselves.
                suffix = suffix[:start] + s["native_target"] + CALL_CLOSE + suffix[end + len(CALL_CLOSE):]
            if a.end_token not in suffix:
                raise ValueError("assistant end token is absent from rendered target")
            completion = suffix.split(a.end_token, 1)[0] + a.end_token
            if reasoning and reasoning not in completion:
                raise ValueError("template dropped teacher reasoning; use a reasoning-capable renderer")
            if calls and a.require_native_roundtrip:
                start, end = completion.find(CALL_OPEN), completion.find(CALL_CLOSE)
                if start < 0 or end < start:
                    raise ValueError(f"{s['id']}: rendered completion has no native call wrapper")
                try:
                    rendered_calls = parse_calls(completion[start:end])
                except (SyntaxError, ValueError) as exc:
                    raise ValueError(f"{s['id']}: rendered native target does not parse: {exc}") from exc
                if rendered_calls != (checked_calls if checked_calls is not None else parsed_target(s)):
                    raise ValueError(f"{s['id']}: rendered completion changed the target call")
        else:
            completion = s["target"]["content"] + a.end_token
        source_groups = s.get("source_groups") or []
        grouped = s.get("family") in ("lambda_scenario", "teacher_program") or s.get("kind") == "review"
        return {"id": s["id"], "program_id": source_groups[0] if grouped and source_groups else program_id(s),
                "source_groups": source_groups, "family": s.get("family", s.get("kind")),
                "skill": s["skill"], "renderer": a.template_id,
                "context_items": len(s["messages"]),
                "teacher_trajectory_id": s.get("teacher_trajectory_id"),
                "teacher_trajectory_digest": s.get("teacher_trajectory_digest"),
                "training_admission": s.get("training_admission"),
                "native_target": s.get("native_target"),
                "prompt": prompt, "completion": completion}, None

    existing = 0
    last_id = None
    if a.resume and a.dst.exists():
        with a.dst.open() as stream:
            for line in stream:
                last_id = json.loads(line)["id"]
                existing += 1

    def pending():
        selected = 0
        matched = existing == 0
        eligible = 0
        terminals = 0
        for line in lines():
            s = json.loads(line)
            if (s.get("provisional_gold") or s.get("template")) and not a.include_template:
                continue
            if eligible % a.every:
                eligible += 1
                continue
            eligible += 1
            if s.get("skill") == "reply":
                keep = terminals % a.terminal_every == 0
                terminals += 1
                if not keep:
                    continue
            if selected >= a.limit:
                break
            if selected < existing:
                if selected == existing - 1:
                    if s["id"] != last_id:
                        raise ValueError("resume destination does not match source at last ID")
                    matched = True
                selected += 1
                continue
            selected += 1
            yield s
        if not matched:
            raise ValueError("resume destination has more rows than source")

    n = existing
    dropped = 0
    drop_reasons = []
    items = iter(pending())
    with ThreadPoolExecutor(max_workers=a.workers) as pool, a.dst.open("a" if a.resume else "w") as out:
        futures = deque()
        for _ in range(a.workers * 4):
            try:
                futures.append(pool.submit(convert, next(items)))
            except StopIteration:
                break
        while futures:
            result, reason = futures.popleft().result()
            if result is None:
                dropped += 1
                if len(drop_reasons) < 50:
                    drop_reasons.append(reason)
            else:
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                n += 1
            if n % 2000 == 0:
                print(n, flush=True)
            try:
                futures.append(pool.submit(convert, next(items)))
            except StopIteration:
                pass
    print(f"{n} pairs, {dropped} invalid legacy actions dropped -> {a.dst}")
    manifest_path.write_text(json.dumps({"renderer": renderer, "source": str(a.src),
                                         "pairs": n, "every": a.every,
                                         "terminal_every": a.terminal_every,
                                         "limit": a.limit,
                                         "dropped_invalid_actions": dropped,
                                         "drop_examples": drop_reasons}, indent=2) + "\n")


if __name__ == "__main__":
    main()
