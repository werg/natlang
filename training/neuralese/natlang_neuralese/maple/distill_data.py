"""Recovery data for a ternary conversion: the BF16 model's OWN behaviour in its own chat template (plans/mellum-port.md,
conversion v3). Conversion v2 recovered on Maple-rendered corpus text with system messages stripped and lost chat
structure (no <think>, no <|im_end|>, looping tool JSON). v3 distills on prompts rendered by the model's tokenizer and
chat template (system prompts and tools kept) plus the BF16 model's sampled responses.

    python -m natlang_neuralese.maple.distill_data prompts --model BF16_DIR --out PROMPTS.jsonl --probe PROBE.jsonl
    python -m natlang_neuralese.maple.distill_data generate --model BF16_DIR --prompts PROMPTS.jsonl \\
        --endpoint http://127.0.0.1:18095 --served-model NAME --out RECORDS.jsonl [--concurrency 24]

A record holds `prompt_ids` (the rendered prompt, generation prompt included) and `response_ids` (the BF16 model's
sampled continuation as token ids); the teacher phase (qat_convert teacher --records) dumps top-k on the whole
sequence and weights prompt positions below response positions.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import sqlite3
import threading
import time
import urllib.request
from pathlib import Path

NATIVE = Path("/home/werg/natlang/runs/luna-v17-actual-quality-proposal-20261008-v9-portable-context-proposal/assembled-v3")
PACKET = Path("/home/werg/natlang/runs/training-periodic-eval-20261003/execution-eval-v3/cases.ir.jsonl")
BIRD = Path("/home/werg/data/bird-sqlite/train/train_databases")
REPO = Path("/home/werg/natlang")
NEURALESE_PARTS = {"read", "neuralese", "digest"}


# Natlang records -------------------------------------------------------------------------------------------------
def _pieces(path: Path) -> dict[str, str]:
    return {p["name"]: p["text"] for p in map(json.loads, open(path))}


def _resolve(content, pieces):
    """Message content with soft prompt pieces replaced by their text; None when it holds Neuralese blocks (a BF16
    model cannot read them)."""
    if not isinstance(content, list):
        return content
    texts = []
    for part in content:
        kind = part.get("type")
        if kind in NEURALESE_PARTS:
            return None
        texts.append(pieces[part["name"]] if kind == "soft" else part.get("text", ""))
    return "".join(texts)


def natlang_prompts(records: Path, pieces: dict, packet_groups: set, split: str = "train"):
    """The context of each natlang training turn (system, user, earlier tool calls and results) with its tools: the
    BF16 model writes the next assistant turn. Records sharing a source group with the protected packet are left out."""
    for row in map(json.loads, open(records)):
        if row.get("split") != split or set(row.get("source_groups") or []) & packet_groups:
            continue
        messages = []
        for message in row["messages"]:
            content = _resolve(message.get("content"), pieces)
            if content is None:
                break
            messages.append({**message, "content": content})
        else:
            yield {"id": row["id"], "source": "natlang:" + records.stem, "messages": messages, "tools": row.get("tools")}


def user_turn_prompts(records: Path, pieces: dict, packet_groups: set):
    """Natlang user turns asked on their own (no system prompt, no tools): plain-chat behaviour on our task text."""
    for row in natlang_prompts(records, pieces, packet_groups):
        user = next((m for m in row["messages"] if m["role"] == "user"), None)
        if user and isinstance(user["content"], str) and 40 < len(user["content"]) < 6000:
            yield {"id": row["id"] + ":user", "source": "user-turn", "messages": [{"role": "user", "content": user["content"]}]}


# Synthetic prompts ---------------------------------------------------------------------------------------------------
TOPICS = ["photosynthesis", "the TCP handshake", "compound interest", "black holes", "garbage collection in Java",
          "the French Revolution", "vaccines", "binary search", "inflation", "plate tectonics", "neural networks",
          "the water cycle", "public-key cryptography", "supply and demand", "DNS resolution", "the immune system",
          "climate change", "database indexing", "the Renaissance", "electric motors", "recursion", "the stock market",
          "CRISPR", "version control with git", "quantum entanglement", "the Roman Empire", "HTTP caching", "sleep",
          "solar panels", "functional programming", "the printing press", "microservices", "volcanoes", "probability",
          "load balancing", "the human heart", "unit testing", "game theory", "the Industrial Revolution", "SQL joins",
          "rainbows", "memory leaks", "the electoral college", "containerization", "bees and pollination", "caching",
          "the Big Bang", "Rust ownership", "coffee brewing", "time zones", "regular expressions", "the moon landing",
          "OAuth", "sourdough bread", "message queues", "the Silk Road", "gradient descent", "event loops in JavaScript",
          "antibiotic resistance", "tides", "the CAP theorem", "chess openings", "API rate limiting", "glaciers"]
GENERAL = ["Explain {t} to a curious twelve-year-old.", "What are the most common misconceptions about {t}?",
           "Give me a three-paragraph overview of {t}.", "Write a short quiz with answers about {t}.",
           "Compare {t} with {u}: what do they have in common?", "Summarize the key ideas of {t} in five bullet points.",
           "Write a friendly email inviting a colleague to a talk about {t}.", "What would an expert want me to know about {t}?",
           "Describe one historical turning point related to {t}.", "List pros and cons of learning about {t} first."]
CODE_TASKS = ["Explain what this code does, step by step.", "Find bugs or edge cases this code mishandles.",
              "Write unit tests for this code.", "Suggest a cleaner refactoring of this code and explain why.",
              "Write a concise docstring or comment block for the main function here.",
              "What are the performance characteristics of this code?", "Rewrite this code in another language of your choice.",
              "Which inputs would make this code fail? Give concrete examples."]
TOOLS = {
    "get_weather": ({"location": {"type": "string", "description": "City name"},
                     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["location"], "Get the current weather for a location."),
    "search_web": ({"query": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"], "Search the web."),
    "calculator": ({"expression": {"type": "string", "description": "Arithmetic expression"}}, ["expression"], "Evaluate an arithmetic expression."),
    "send_email": ({"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, ["to", "subject", "body"], "Send an email."),
    "create_event": ({"title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"},
                      "time": {"type": "string"}}, ["title", "date"], "Create a calendar event."),
    "read_file": ({"path": {"type": "string"}}, ["path"], "Read a text file."),
    "run_sql": ({"query": {"type": "string"}}, ["query"], "Run a read-only SQL query and return rows."),
    "get_stock_price": ({"ticker": {"type": "string"}}, ["ticker"], "Get the latest stock price."),
    "convert_currency": ({"amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}},
                         ["amount", "from", "to"], "Convert an amount between currencies."),
    "set_reminder": ({"text": {"type": "string"}, "minutes": {"type": "integer"}}, ["text", "minutes"], "Set a reminder."),
    "translate": ({"text": {"type": "string"}, "target_language": {"type": "string"}}, ["text", "target_language"], "Translate text."),
}
TOOL_REQUESTS = {
    "get_weather": ["What's the weather like in {city} right now?", "Should I bring an umbrella in {city} today?"],
    "search_web": ["Find recent news about {thing}.", "Look up who invented {thing}."],
    "calculator": ["What is {a} times {b} plus {c}?", "Compute ({a} + {b}) / {c} for me."],
    "send_email": ["Email {person} at {person}@example.com that the meeting moves to {day}.",
                   "Send a thank-you note to {person}@example.com about the {thing} demo."],
    "create_event": ["Put a dentist appointment on 2026-11-{dd} at 3pm in my calendar.", "Schedule '{thing} review' for 2026-12-{dd}."],
    "read_file": ["Open notes/{thing}.txt and tell me what it says.", "What's in config/{thing}.yaml?"],
    "run_sql": ["How many orders were placed in {month}? The table is orders(id, placed_at, total).",
                "List the five largest customers by revenue from customers(id, name) and orders(customer_id, total)."],
    "get_stock_price": ["What is {ticker} trading at?", "Is {ticker} up or down today?"],
    "convert_currency": ["How much is {a} USD in EUR?", "Convert {a} JPY to GBP."],
    "set_reminder": ["Remind me to call {person} in {a} minutes.", "Set a reminder to stretch in {c} minutes."],
    "translate": ["Translate 'where is the train station' into {lang}.", "How do you say 'thank you for your help' in {lang}?"],
}
FILL = {"city": ["Berlin", "Lagos", "Lima", "Osaka", "Toronto", "Cairo", "Oslo", "Mumbai"], "person": ["ana", "kofi", "li", "marta", "sam"],
        "thing": ["the transistor", "rust compilers", "solar batteries", "budget", "launch", "the telescope", "kubernetes"],
        "day": ["Monday", "Thursday", "Friday"], "month": ["March 2026", "June 2026"], "ticker": ["AAPL", "NVDA", "SAP", "TSLA"],
        "lang": ["Spanish", "Japanese", "German", "Swahili", "Portuguese"]}


def _tool_schema(name):
    properties, required, description = TOOLS[name]
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": properties, "required": required}}}


def _fill(template, rng):
    values = {k: rng.choice(v) for k, v in FILL.items()}
    values.update(a=rng.randint(2, 999), b=rng.randint(2, 99), c=rng.randint(2, 60), dd=f"{rng.randint(1, 28):02d}")
    return template.format(**values)


def tool_prompt(rng, index):
    name = rng.choice(sorted(TOOLS))
    others = rng.sample(sorted(set(TOOLS) - {name}), rng.randint(0, 3))
    tools = [_tool_schema(t) for t in rng.sample([name] + others, len(others) + 1)]
    messages = [{"role": "user", "content": _fill(rng.choice(TOOL_REQUESTS[name]), rng)}]
    if rng.random() < 0.15:
        messages.insert(0, {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when needed."})
    return {"id": f"tool:{index}", "source": "tool-call", "messages": messages, "tools": tools, "expected_tool": name}


def math_prompt(rng, index):
    a, b, c = rng.randint(12, 999), rng.randint(3, 97), rng.randint(2, 40)
    kind = rng.randrange(6)
    if kind == 0:
        q, ans = f"What is {a} + {b} * {c}?", a + b * c
    elif kind == 1:
        q, ans = f"A shop sells pens at {b} cents each. How many cents do {c} pens cost?", b * c
    elif kind == 2:
        q, ans = f"What is {c}% of {a * 100}?", c * a
    elif kind == 3:
        q, ans = f"I had {a} marbles, gave away {b}, then received {c}. How many do I have now?", a - b + c
    elif kind == 4:
        q, ans = f"What is the remainder when {a * b + c % b} is divided by {b}?", (a * b + c % b) % b
    else:
        q, ans = f"A train travels {b * c} km in {c} hours at constant speed. What is its speed in km/h?", b
    if rng.random() < 0.5:
        q += " Answer with just the number."
    return {"id": f"math:{index}", "source": "math", "messages": [{"role": "user", "content": q}], "answer": str(ans)}


def general_prompt(rng, index):
    t, u = rng.sample(TOPICS, 2)
    return {"id": f"general:{index}", "source": "general",
            "messages": [{"role": "user", "content": rng.choice(GENERAL).format(t=t, u=u)}]}


def code_prompt(rng, index, files):
    path = rng.choice(files)
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 12:
        return None
    start = rng.randrange(0, max(1, len(lines) - 40))
    snippet = "\n".join(lines[start:start + rng.randint(15, 45)])
    language = "python" if path.suffix == ".py" else "typescript"
    return {"id": f"code:{index}", "source": "code",
            "messages": [{"role": "user", "content": f"{rng.choice(CODE_TASKS)}\n\n```{language}\n{snippet}\n```"}]}


def sql_prompts(rng, count):
    databases = sorted(BIRD.glob("*/*.sqlite"))
    out = []
    for index in range(count):
        db = rng.choice(databases)
        try:
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
                rows = con.execute("select name, sql from sqlite_master where type='table' and sql is not null").fetchall()
        except sqlite3.Error:
            continue
        if not rows:
            continue
        schema = "\n".join(sql for _, sql in rows)[:4000]
        table = rng.choice(rows)[0]
        ask = rng.choice([f"How many rows does {table} have?", f"Which values occur most often in a text column of {table}?",
                          f"Write a query that joins {table} with a related table and returns ten rows.",
                          f"Find duplicate entries in {table}.", f"Give summary statistics for a numeric column of {table}."])
        out.append({"id": f"sql:{db.stem}:{index}", "source": "sql",
                    "messages": [{"role": "user", "content": f"SQLite schema:\n{schema}\n\nWrite a SQLite query: {ask}"}]})
    return out


def build(seed: int, counts: dict):
    rng = random.Random(seed)
    pieces = _pieces(NATIVE / "native-pieces.jsonl") | _pieces(NATIVE / "recurrence-pieces.jsonl")
    groups = set()
    for case in map(json.loads, open(PACKET)):
        groups.update(case.get("source_groups") or [])
    prompts = []
    natlang = list(natlang_prompts(NATIVE / "native-records.jsonl", pieces, groups))
    seen = {p["id"] for p in natlang}
    recurrence = [p for p in natlang_prompts(NATIVE / "recurrence-records.jsonl", pieces, groups) if p["id"] not in seen]
    rng.shuffle(recurrence)
    prompts += natlang[:counts["natlang"]] + recurrence[:counts["recurrence"]]
    users = list(user_turn_prompts(NATIVE / "native-records.jsonl", pieces, groups))
    rng.shuffle(users)
    prompts += users[:counts["user"]]
    files = sorted(p for p in (REPO / "training/neuralese/natlang_neuralese").rglob("*.py")) + \
        sorted((REPO / "ts-host/src").rglob("*.ts"))
    code = [p for p in (code_prompt(rng, i, files) for i in range(counts["code"])) if p]
    prompts += code + [math_prompt(rng, i) for i in range(counts["math"])]
    prompts += [general_prompt(rng, i) for i in range(counts["general"])]
    prompts += [tool_prompt(rng, i) for i in range(counts["tool"])] + sql_prompts(rng, counts["sql"])
    for prompt in prompts:  # thinking on for about half; natlang turns as the harness runs them (both modes)
        prompt["enable_thinking"] = rng.random() < 0.5
    rng.shuffle(prompts)
    return prompts


def probe_set(seed: int):
    """24 fixed held-out prompts for the generation gate: plain exact answers (thinking off), reasoning with exact
    answers (thinking on), tool calls with a schema, natlang harness-style turns (test split)."""
    rng = random.Random(seed + 7919)
    probes = []
    for i in range(6):
        p = math_prompt(rng, 10_000 + i)
        probes.append({**p, "id": f"probe:plain:{i}", "kind": "plain", "enable_thinking": False})
    for i in range(6):
        p = math_prompt(rng, 20_000 + i)
        probes.append({**p, "id": f"probe:thinking:{i}", "kind": "thinking", "enable_thinking": True})
    for i in range(6):
        p = tool_prompt(rng, 30_000 + i)
        probes.append({**p, "id": f"probe:tool:{i}", "kind": "tool", "enable_thinking": i % 2 == 0})
    pieces = _pieces(NATIVE / "native-pieces.jsonl")
    groups = set()
    for case in map(json.loads, open(PACKET)):
        groups.update(case.get("source_groups") or [])
    natlang = list(natlang_prompts(NATIVE / "native-records.jsonl", pieces, groups, split="test"))
    rng.shuffle(natlang)
    for i, p in enumerate(natlang[:6]):
        probes.append({**p, "id": f"probe:natlang:{i}", "kind": "natlang", "enable_thinking": i % 2 == 0})
    return probes


# Rendering and generation ----------------------------------------------------------------------------------------------
def render(tokenizer, prompt) -> list[int]:
    return tokenizer.apply_chat_template(prompt["messages"], tools=prompt.get("tools") or None, tokenize=True,
                                         add_generation_prompt=True, enable_thinking=bool(prompt.get("enable_thinking")),
                                         return_dict=False)


def _post(url, body, timeout=1800):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def generate(a):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    prompts = [json.loads(l) for l in open(a.prompts)]
    out = Path(a.out)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    lock, written, started = threading.Lock(), [0], time.time()
    handle = open(out, "a")

    def one(prompt):
        ids = render(tokenizer, prompt)
        if len(ids) > a.max_prompt_tokens:
            return None
        limit = a.max_tokens_thinking if prompt.get("enable_thinking") else a.max_tokens
        seed = int(hashlib.sha256(prompt["id"].encode()).hexdigest()[:8], 16)
        body = {"model": a.served_model, "prompt": ids, "max_tokens": limit, "temperature": a.temperature,
                "top_p": a.top_p, "seed": seed, "logprobs": 0, "return_tokens_as_token_ids": True,
                "skip_special_tokens": False}
        result = _post(a.endpoint + "/v1/completions", body)["choices"][0]
        tokens = result["logprobs"]["tokens"]
        response = [int(t.split(":", 1)[1]) for t in tokens]
        return {"id": prompt["id"], "source": prompt["source"], "enable_thinking": bool(prompt.get("enable_thinking")),
                "prompt_ids": ids, "response_ids": response, "finish_reason": result.get("finish_reason"),
                **{k: prompt[k] for k in ("answer", "expected_tool") if k in prompt}}

    pending = [p for p in prompts if p["id"] not in done]
    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
        for record in pool.map(lambda p: _safe(one, p), pending):
            if record is None:
                continue
            with lock:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                written[0] += 1
                if written[0] % 100 == 0:
                    print(json.dumps({"written": written[0] + len(done), "of": len(prompts),
                                      "per_minute": round(written[0] / ((time.time() - started) / 60), 1)}), flush=True)


def _safe(fn, prompt):
    for attempt in range(3):
        try:
            return fn(prompt)
        except Exception as error:  # noqa: BLE001 — a failed request is retried, then skipped and reported
            last = error
            time.sleep(5 * (attempt + 1))
    print(json.dumps({"failed": prompt["id"], "error": str(last)[:200]}), flush=True)
    return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    b = sub.add_parser("prompts")
    b.add_argument("--out", required=True)
    b.add_argument("--probe", required=True)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--counts", default="natlang=2600,recurrence=1600,user=600,code=900,math=700,general=700,tool=900,sql=400")
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--prompts", required=True)
    g.add_argument("--endpoint", required=True)
    g.add_argument("--served-model", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--concurrency", type=int, default=24)
    g.add_argument("--max-prompt-tokens", type=int, default=7168)
    g.add_argument("--max-tokens", type=int, default=1024)
    g.add_argument("--max-tokens-thinking", type=int, default=2048)
    g.add_argument("--temperature", type=float, default=0.6)
    g.add_argument("--top-p", type=float, default=0.95)
    a = p.parse_args(argv)
    if a.action == "prompts":
        counts = {k: int(v) for k, v in (item.split("=") for item in a.counts.split(","))}
        prompts = build(a.seed, counts)
        with open(a.out, "w") as handle:
            for prompt in prompts:
                handle.write(json.dumps(prompt) + "\n")
        with open(a.probe, "w") as handle:
            for prompt in probe_set(a.seed):
                handle.write(json.dumps(prompt) + "\n")
        sources = {}
        for prompt in prompts:
            sources[prompt["source"]] = sources.get(prompt["source"], 0) + 1
        print(json.dumps({"prompts": len(prompts), "sources": sources}))
    else:
        generate(a)


if __name__ == "__main__":
    main()
