"""Trajectory training on Neuralese-converted natlang records (DECISIONS.md 40, 41; S5 §3, first step).

Input: records written by ts-host/scripts/neuralese-convert-trajectories.mjs and its pieces file. Every `soft` part
(the runtime's prompt pieces, older system-prompt versions, program guidance) is a trainable soft parameter,
initialised from its text: from a system-prompt bank where one is given and the piece is in it (`prompt:<id>`),
otherwise encoded from the text in one pass through the port (`encode_text`). Each training step teacher-forces the record's target turn (cross-entropy on
its tokens) with the soft parameters as gradient leaves, optionally with a LoRA on the backbone. The trained
parameters are saved as a bank of the current runtime's pieces (`system-prompts.nz`) and as all soft parameters by
name (`soft-params.pt`).

Written values (`--handover written`, `--digest written`): handover notes, child calls' results and listing digests are written by the
model through the port's differentiable write procedure (S3 `unroll_write`), afresh at every step, and enter their
readers as gradient leaves. The readers' losses therefore train the writer: the payload carries gradients into the
content projection, the sketch recurrence, the LoRA and every soft parameter of the write site (the digest
instructions, the producing record's prompts), so a note or digest learns to hold what its readers need. A note is
written from its producing record (the record whose target is the `compact_history` call), soft-rendered, at the note
argument (the reply forced to the model's own rendering of `compact_history(note='`, chat.call_reply, then the
write: the same cut the server's template readout makes). A child call's result (converter: child results) is
written the same way from the child's final record at `return_result(status='success', value='` and read where the
caller's eval output shows it: the recurrence of calling a function, retrieving its value and splicing it into the
caller's trajectory, trained across the run's chunks (the child's record and every caller record that reads it) in
one graph. Writes nest: a producer's own context reads the values it was given written afresh too, to
`--write-depth` levels (a caller reads a child's result whose child read a grandchild's); deeper ones are crisp. A
digest by the digest operator's plan
(digest.py), chunked when the value exceeds `--digest-window` tokens, every chunk write and the combining write
differentiable. With `--tokens-per-vector R` a write is sized from the crisp text it stands for (the note's text, the
listing preview a digest replaces): ceil(tokens / R) vectors, no stop decision, and the stop head is trained on that
boundary (`--stop-weight`). This is the simple training regime; sizes are never required at inference, where the stop
head decides unless a caller passes a length hint. Otherwise stop decisions are sampled and trained by a policy gradient with reward −(reader loss + λ·length)
against a running baseline (`--stop-pg λ`); without it they are detached and the length is the stop head's choice. With the crisp modes, notes are rendered as the crisp note and
digests as the listing's preview. Records that read written values add a self-distillation term (weight `--distill`)
from the same model given the crisp note and preview.

Evaluation on held-out records (split `test`): mean target cross-entropy with the system prompt as crisp text, with the
soft parameters as initialised, and as trained. With written values, the held-out readers are also scored with the
values written for them against values written for another reader of different values (`written-init`,
`written-trained`: written, shuffled, the fraction where written is better); `-train` on the first --eval training
readers (whether readers use the written content at all, when few held-out readers fit).

Usage: python -m natlang_neuralese.train.trajectories --records converted.jsonl --pieces pieces.jsonl --out DIR
         [--heads CKPT --bank system-prompts.nz --steps 500 --batch 4 --lr 1e-3 --rank 0 --lora-lr 2e-4
          --max-tokens 6144 --train 2000 --eval 64]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import random
import re
import signal
import os
import gc
import time
from pathlib import Path

import torch

from ..digest import PREFIX as DIGEST_PREFIX, digest_note, write_digest

INSTRUCTIONS = re.compile(r"Instructions:\n([\s\S]*?)\n\n(?:In eval|Eval also|$)")


def crisp_messages(messages: list[dict], texts: dict[str, str], notes: dict[str, str]) -> list[dict]:
    """Messages with every soft part as its text and every handover as its note: the crisp rendering."""
    return render(messages, lambda name: {"type": "text", "text": texts[name]}, notes)


def render(messages: list[dict], soft_part, notes: dict[str, str], blocks: dict[str, str] | None = None,
           digests: dict[str, str] | None = None) -> list[dict]:
    """Converted messages → engine messages: `soft` parts via `soft_part(name)`; handover reads and writes as the
    written block where `blocks` has one (name → block ID), else as the crisp note; parts merged into text where no
    block remains."""
    blocks, digests = blocks or {}, digests or {}
    out = []
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part["type"] == "soft":
                    parts.append(soft_part(part["name"]))
                elif part["type"] == "digest":
                    # Written digests (decision 43) show as the runtime lists them; otherwise the crisp cut-off preview.
                    if part["name"] in digests:
                        parts.append({"type": "neuralese", "id": digests[part["name"]]})
                        parts.append({"type": "text", "text": digest_note(part["holder"])})
                    else:
                        parts.append({"type": "text", "text": part["preview"]})
                elif part["type"] == "read":
                    name = part["name"]
                    parts.append({"type": "neuralese", "id": blocks[name]} if name in blocks else
                                 {"type": "text", "text": part.get("source", notes.get(name, ""))})
                else:
                    parts.append({"type": "text", "text": part["text"]})
            message["content"] = ("".join(p["text"] for p in parts) if all(p["type"] == "text" for p in parts) else parts)
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                args = call["function"]["arguments"]
                if '"$write"' in args:
                    value = json.loads(args)
                    written = {k: v["$write"]["name"] for k, v in value.items() if isinstance(v, dict) and "$write" in v and v["$write"]["name"] in blocks}
                    def crisp_value(v):
                        if not isinstance(v, dict) or "$write" not in v:
                            return v
                        site = v["$write"]
                        return json.loads(site["source"]) if site.get("type") == "Neuralese<unknown>" else site["source"]
                    value = {k: crisp_value(v) for k, v in value.items()}
                    if written:
                        # Preserve the declared argument order and typed-value
                        # boundary; rendering handles the opaque wire marker.
                        original = json.loads(args)
                        value = {key: ([{"type": "neuralese", "id": blocks[written[key]],
                                          "value_type": "string" if item["$write"].get("type", "Neuralese<string>") == "Neuralese<string>" else "unknown"}]
                                       if key in written else crisp_value(item))
                                 for key, item in original.items()}
                        arguments = json.dumps(value, ensure_ascii=False)
                    else:
                        arguments = json.dumps(value, ensure_ascii=False)
                    call = {**call, "function": {**call["function"], "arguments": arguments}}
                calls.append(call)
            message["tool_calls"] = calls
        out.append(message)
    return out


def reads(record: dict) -> set[str]:
    return {part["name"] for m in record["messages"] if isinstance(m.get("content"), list)
            for part in m["content"] if part["type"] == "read"}


def write_site(record: dict) -> tuple[str, dict, str, str] | None:
    """Where a record's target writes a value: (call name, the arguments before the written one, that argument's
    name, the value's name). A compaction call writes its note (`compact_history(note=…)`); a child call's final
    turn writes its result (`return_result(status='success', value=…)`, the template readout's site)."""
    for call in (record.get("target") or {}).get("tool_calls") or []:
        if '"$write"' in call["function"]["arguments"]:
            arguments = json.loads(call["function"]["arguments"])
            before = {}
            for key, value in arguments.items():
                if isinstance(value, dict) and "$write" in value:
                    return call["function"]["name"], before, key, value["$write"]["name"]
                before[key] = value
    return None


def write_value_type(record: dict) -> str:
    for call in (record.get("target") or {}).get("tool_calls") or []:
        for value in json.loads(call["function"]["arguments"]).values():
            if isinstance(value, dict) and "$write" in value:
                return "string" if value["$write"].get("type", "Neuralese<string>") == "Neuralese<string>" else "unknown"
    return "string"


def target_write(record: dict) -> str | None:
    """The name of the value a record's target writes (a handover note, a child call's result), if it writes one."""
    site = write_site(record)
    return site[3] if site else None


def handover_notes(record: dict) -> dict[str, str]:
    """Note texts by name: from the record's compaction calls and from its reads (which carry their note)."""
    notes = {part["name"]: part["source"] for m in record["messages"] if isinstance(m.get("content"), list)
             for part in m["content"] if part["type"] == "read" and "source" in part}
    for message in record["messages"] + ([record["target"]] if record.get("target") else []):
        for call in message.get("tool_calls") or []:
            if '"$write"' in call["function"]["arguments"]:
                for value in json.loads(call["function"]["arguments"]).values():
                    if isinstance(value, dict) and "$write" in value:
                        notes[value["$write"]["name"]] = value["$write"]["source"]
    return notes


def producer_text_target(record, texts, names):
    """Gold ordinary reply at a producer, preserving its incoming soft values.

    Its own output must stay gold text here, so body-token supervision is not
    replaced by the generated opaque payload that the reader objective consumes.
    """
    own = target_write(record)
    ancestors = {name: block for name, block in names.items() if name != own}
    return render([record['target']], lambda name: {'type': 'text', 'text': texts[name]},
                  handover_notes(record), ancestors)[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", required=True)
    parser.add_argument("--pieces", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--crisp-weight", type=float, default=0.0, help="additional ordinary-text SFT, backward separately before the same optimizer step; preserves interpreter policy alongside soft-return learning")
    parser.add_argument("--writer-text-weight", type=float, default=None, help="teacher-forced gold producer reply under its actual soft/ancestor context; additional local writer objective")
    parser.add_argument("--curriculum-change", action="append", default=[], choices=["tokens_per_vector", "writer_text_weight", "write_depth", "write_curriculum", "max_writes"], help="explicitly permit named curriculum changes at --continue-from while preserving optimizer/RNG and fixed data")
    parser.add_argument("--continue-from", help="explicit new code stage preserving full optimizer/RNG; requires identical data and training controls")
    parser.add_argument("--soft-init", help="warm-start matching soft parameters from a prior soft-params or full recurrence checkpoint; new pieces are text-initialized")
    parser.add_argument("--heads", default=None, help="port heads checkpoint (soft parameters are read through them)")
    parser.add_argument("--base", default=None)
    parser.add_argument("--bank", default=None, help="system-prompt bank to initialise current pieces from")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=None,
                        help="records/chains per optimizer step (default: 1 for sampled-chain, 4 for joint)")
    parser.add_argument("--lr", type=float, default=1e-3, help="soft parameters")
    parser.add_argument("--rank", type=int, default=0, help="LoRA rank (0: soft parameters only)")
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--max-tokens", type=int, default=6144, help="skip records whose crisp prompt is longer")
    parser.add_argument("--train", type=int, default=2000, help="training records to read")
    parser.add_argument("--eval", type=int, default=64, help="held-out records")
    parser.add_argument("--handover", choices=["crisp", "written"], default="crisp",
                        help="handoffs (handover notes, child calls' results): their crisp text, or written by their producer")
    parser.add_argument('--write-curriculum', choices=['joint', 'sampled-chain'], default='joint',
                        help='joint uses all handoffs unless max-writes is set; sampled-chain picks one handoff per nested level each step')
    parser.add_argument("--max-writes", type=int, default=0,
                        help="write at most this many of a record's handoffs per step, chosen at random each step; the rest "
                             "read their crisp text (bounds memory: each written value prefills its producer with gradient; 0: all)")
    parser.add_argument("--write-depth", type=int, default=2,
                        help="levels of written values inside written values' producers (1: producers read crisp text)")
    parser.add_argument("--only-handover", action="store_true", help="only records that read or write a note")
    parser.add_argument("--digest", choices=["preview", "written"], default="preview",
                        help="digest sites: the crisp preview, or a digest the model writes at the operator's write site")
    parser.add_argument("--digest-window", type=int, default=4096,
                        help="value tokens per digest write site in training (longer values are digested in chunks)")
    parser.add_argument("--stop-pg", type=float, default=0.0,
                        help="train the stop head on written values by policy gradient with this length cost per vector (0: off)")
    parser.add_argument("--tokens-per-vector", type=float, default=0.0,
                        help="size each written value from the crisp text it stands for (the note's text, the digest's listing "
                             "preview): ceil(tokens / this) vectors, the stop head trained on that boundary (0: the stop head decides)")
    parser.add_argument("--stop-weight", type=float, default=1.0, help="weight of the stop-boundary loss on source-sized writes")
    parser.add_argument("--heads-lr", type=float, default=1e-4, help="the writer's port heads, when notes or digests are written")
    parser.add_argument("--detach-write-context", action="store_true",
                        help="no gradient into the write sites' context (saves memory; soft prompts there then do not learn from writing)")
    parser.add_argument("--distill", type=float, default=1.0, help="weight of the self-distillation term on written notes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-gb", type=float, default=None,
                        help="CUDA envelope (default: ledger NATLANG_CUDA_MEMORY_GB, else 90%% of free CUDA memory)")
    parser.add_argument('--backward-policy', choices=['joint', 'staged', 'auto'], default='joint')
    parser.add_argument('--graph-memory-gb', type=float, default=0,
                        help='auto joint graph high-water budget; 0 derives from the CUDA envelope')
    parser.add_argument('--graph-headroom-gb', type=float, default=.35,
                        help='reserve within CUDA envelope for backward temporaries and optimizer state')
    parser.add_argument("--activation-offload-gb", type=float, default=0,
                        help="CPU budget for saved activations; exact gradients, no recomputation or detached writes")
    parser.add_argument('--checkpoint-layers', action='store_true',
                        help='recompute layer activations during backward to reduce memory; preserves full recurrence gradients')
    parser.add_argument('--ffn-chunk-tokens', type=int, default=0,
                        help='token-local FFN chunks reduce transient allocations without context truncation')
    parser.add_argument('--optimizer', choices=['adamw', 'muon'], default='adamw')
    parser.add_argument('--checkpoint-every', type=int, default=25)
    parser.add_argument('--eval-every', type=int, default=0, help='periodic held-out soft and written-vs-shuffled probes; 0: initial/final only')
    args = parser.parse_args(argv)
    from .staging import StagedWrites, resolve_values, GraphBudgetExceeded, graph_memory_budget
    from .recurrence import curriculum_max_writes
    args.max_writes = curriculum_max_writes(args.write_curriculum, args.max_writes)
    if args.batch is None:
        args.batch = 1 if args.write_curriculum == 'sampled-chain' else 4
    if args.steps < 1 or args.batch < 1 or args.checkpoint_every < 1 or args.write_depth < 1 or args.activation_offload_gb < 0 or args.ffn_chunk_tokens < 0 or args.eval_every < 0:
        raise ValueError('invalid recurrence training controls')
    if args.curriculum_change and not args.continue_from:
        raise ValueError('curriculum changes require explicit continuation checkpoint')
    if args.writer_text_weight is not None and (not math.isfinite(args.writer_text_weight) or args.writer_text_weight < 0):
        raise ValueError('invalid producer text supervision weight')
    if not math.isfinite(args.crisp_weight) or args.crisp_weight < 0:
        raise ValueError('invalid crisp SFT weight')
    if args.graph_memory_gb < 0 or args.graph_headroom_gb <= 0:
        raise ValueError('invalid graph memory budget')
    if args.backward_policy != 'joint' and (args.stop_pg or args.digest == 'written' or args.activation_offload_gb):
        raise ValueError('staging currently requires deterministic handoffs, preview digests and zero CPU offload')
    if args.device.startswith('cuda'):
        free, total = torch.cuda.mem_get_info()
        args.memory_gb = args.memory_gb or float(os.environ.get('NATLANG_CUDA_MEMORY_GB') or free / 2**30 * .9)
        envelope = min(args.memory_gb, total / 2**30)
        graph_budget = (args.graph_memory_gb or envelope - args.graph_headroom_gb) * 2**30
        if graph_budget <= 0 or graph_budget >= envelope * 2**30:
            raise ValueError('graph budget must leave backward headroom within the CUDA envelope')
    else:
        graph_budget = 0
        args.memory_gb = args.memory_gb or 8

    from ..prompt_bank import load_bank, save_bank
    from ..serve import load_engine
    from ..serve.chat import RequestError, call_reply, write_reply, render_messages
    from ..serve.grad import GradSession, encode_text
    from .execution import Prefilled, unroll_write
    from .losses import stop_boundary_loss
    from ..serve.store import make_block
    from .adapters import inject_lora, lora_state

    out = Path(args.out)
    checkpoint_path = out / 'checkpoint.pt'
    if out.exists() and any(out.iterdir()) and not checkpoint_path.exists():
        raise ValueError('existing recurrence output has no resumable checkpoint; use a fresh directory')
    out.mkdir(parents=True, exist_ok=True)
    from .trajectory_state import atomic_checkpoint, trajectory_optimizer, validate_resume, validate_continuation, gradient_norm, clip_finite_gradients
    def digest_file(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b''):
                digest.update(chunk)
        return digest.hexdigest()
    identity = {'options': {k: v for k, v in vars(args).items() if k not in {'out', 'memory_gb', 'activation_offload_gb', 'checkpoint_every', 'backward_policy', 'graph_memory_gb', 'graph_headroom_gb', 'continue_from', 'curriculum_change'} and not (k == 'writer_text_weight' and v is None)},
                'files': {str(Path(p).resolve()): digest_file(p) for p in [args.records, args.pieces, args.heads, args.bank, args.soft_init] if p},
                'code': {str(p.resolve()): digest_file(p) for p in Path(__file__).resolve().parents[1].rglob('*.py')}}
    if args.continue_from:
        identity['continuation'] = {'checkpoint_sha256': digest_file(args.continue_from),
                                    'path': str(Path(args.continue_from).resolve()),
                                    'curriculum_changes': args.curriculum_change}
    resumed = torch.load(checkpoint_path, map_location='cpu', weights_only=False) if checkpoint_path.exists() else None
    if resumed is not None:
        validate_resume(resumed, identity)
    elif args.continue_from:
        resumed = torch.load(args.continue_from, map_location='cpu', weights_only=False)
        validate_continuation(resumed, identity, allowed_changes=args.curriculum_change)
    if args.device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_gb * 2**30 / total))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    engine = load_engine(args.base, heads_checkpoint=args.heads, device=args.device)
    if not engine.heads.read_markers:
        foundation = getattr(engine, 'foundation', None) or {}
        if not foundation.get('qualified') or not foundation.get('runtime_qualified'):
            raise ValueError('raw neuralese recurrence requires a certified, runtime-qualified foundation handoff')
    for p in engine.backbone.parameters():
        p.requires_grad_(False)
    session = GradSession(engine)

    texts = {}
    with open(args.pieces) as stream:
        for line in stream:
            piece = json.loads(line)
            texts[piece["name"]] = piece["text"]
    bank = load_bank(args.bank) if args.bank else None
    params, leaf_ids, from_bank = {}, {}, []

    producers = {}
    if args.handover == "written":
        # Every record whose target writes a note, by note name: the producer of that note's block.
        with open(args.records) as stream:
            for line in stream:
                if "$write" in line:  # escaped inside the arguments string in the raw line
                    record = json.loads(line)
                    name = target_write(record)
                    if name:
                        producers.setdefault(name, record)

    backbone, heads = engine.backbone, engine.heads
    backbone.ffn_chunk_tokens = args.ffn_chunk_tokens
    backbone.checkpoint_layers = args.checkpoint_layers
    active_staging = [None]
    from .memory_estimator import AdaptiveGraphMemory, geometry_bytes
    memory_estimator = AdaptiveGraphMemory(resumed.get('memory_estimator') if resumed else None)
    memory_layout = dict(width=backbone.config.hidden_size, layers=backbone.num_layers,
                         intermediate=backbone.layers[0].feed_forward.w1.out_features,
                         kv_width=sum(2 * backbone.layers[i].self_attn.k_proj.out_features
                                      for i in range(backbone.num_layers) if backbone.is_attention(i)),
                         dtype_bytes=backbone.embedding_weight.element_size(), checkpointed=args.checkpoint_layers)
    geometry_cache = {}
    from .recurrence import ProducerMemo, is_acyclic
    dependencies = {name: (reads(record) | set(handover_notes(record))) - {name}
                    for name, record in producers.items()}
    share_producers = is_acyclic(dependencies) and not args.max_writes and not args.stop_pg

    def placeholder(name: str) -> str:
        """A block ID that stands for a value written afresh each step; its rows always come from the leaves."""
        digest = hashlib.sha256(name.encode()).digest()
        return "nz1_" + base64.b32encode(digest).decode().rstrip("=").lower()

    def source_length(text: str | None) -> int | None:
        """The supervised length of a write standing for `text`, or None (the stop head decides)."""
        if not args.tokens_per_vector or not text:
            return None
        return max(1, min(heads.max_length, math.ceil(len(engine._tokens(text)) / args.tokens_per_vector)))

    def write(messages, tools, prefix, leaves, source: str | None = None):
        """The write procedure with gradients (S3 `unroll_write`): the site's prompt (soft parts from `leaves`), the
        forced prefix and the open marker, then the sketch recurrence until the stop head stops. The stop decisions
        are detached; the payload carries gradients into the writer (feedback, content projection, LoRA) and into
        every soft parameter of the site."""
        prompt = render_messages(messages, tools, engine._template, engine.specials)
        items = session._items(prompt.segments, prompt.blocks, prompt.escape_nonce) + [("tok", t) for t in engine._tokens(prefix)]
        context = session._embed_items(items, leaves)
        write_context_lengths.append(context.shape[1])
        if args.detach_write_context:
            context = context.detach()
        try:
            from .execution import prefill_write_context
            pre = prefill_write_context(backbone, heads, context)
        except torch.OutOfMemoryError:
            print(json.dumps({'status': 'producer_out_of_memory', 'context_tokens': context.shape[1],
                              'ffn_chunk_tokens': args.ffn_chunk_tokens}), flush=True)
            raise
        target = source_length(source)
        if target is not None:
            # Sized from the crisp text it stands for: no stop decision; the stop head learns the boundary.
            written = unroll_write(backbone, heads, pre, length=target)
            if args.stop_weight and torch.is_grad_enabled():
                boundary_terms.append(args.stop_weight * stop_boundary_loss(written))
        else:
            written = unroll_write(backbone, heads, pre, sample=bool(args.stop_pg), generator=stop_generator)
        n = int(written.lengths[0])
        lengths.append(n)
        if target is None and args.stop_pg and torch.is_grad_enabled():
            # Log-probability of the sampled stop decisions under the stop head (continue after 1..n-1, stop after n
            # unless the write ran to the maximum), differentiable in the stop head.
            logits = written.stop_logits[0].float()
            logp = torch.nn.functional.logsigmoid(-logits[:n - 1]).sum()
            if not bool(written.truncated[0]):
                logp = logp + torch.nn.functional.logsigmoid(logits[n - 1])
            stop_terms.append((logp, n))
        return written.payload[0, :n]

    lengths: list[int] = []
    write_context_lengths: list[int] = []
    stop_terms: list = []  # (log-probability of the stop decisions, length) of this record's writes
    boundary_terms: list = []  # stop-boundary losses of this record's source-sized writes
    stop_generator = torch.Generator().manual_seed(args.seed)
    write_choice = random.Random(args.seed)
    baseline = {"value": None}

    prefixes = {}

    def site_prefix(record):
        """The forced reply before the written argument, in the model's own rendering of the producer's call."""
        call, before, argument, _ = write_site(record)
        value_type = write_value_type(record) if engine.heads.profile == "raw-token-v1" else "string"
        key = (call, json.dumps(before, sort_keys=True), argument, value_type)
        if key not in prefixes:
            prefixes[key] = write_reply(lambda m, g: engine.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=g),
                                       call, before, argument, value_type)[0]
        return prefixes[key]

    def note_payload(name, leaves, depth=1, visiting=(), memo=None):
        """The value (a note, a child's result) written by the model from its producing record (soft-rendered), at the
        written argument. The producer's own reads are written afresh too, to --write-depth levels."""
        if memo is None:
            memo = ProducerMemo(share_producers)
        def compute():
            producer = producers[name]
            names, payloads = {}, {}
            if depth < args.write_depth:
                # Depth counts writes, once per edge. Previously incrementing here
                # AND in written_values silently made depth3 only two write layers.
                names, payloads = written_values(producer, leaves, depth, visiting + (name,), memo)
            messages = render(producer["messages"], lambda n: {"type": "neuralese", "id": leaf_ids[n]}, handover_notes(producer),
                              names, names)
            def replay():
                begin = len(boundary_terms)
                result = write(messages, producer.get("tools"), site_prefix(producer),
                               resolve_values({**leaves, **payloads}), source=handover_notes(producer).get(name))
                terms = boundary_terms[begin:]
                del boundary_terms[begin:]
                return result, terms
            def gold_replay():
                if producer.get('split') != 'train' or producer.get('training_admission', {}).get('approved') is not True:
                    raise ValueError('producer gold supervision requires admitted training split')
                target = producer_text_target(producer, texts, names)
                return args.writer_text_weight * session.supervised_text_loss(
                    {'messages': messages, 'tools': producer.get('tools'), 'target': target},
                    resolve_values({**leaves, **payloads}))
            auxiliary = gold_replay if args.writer_text_weight and torch.is_grad_enabled() else None
            if active_staging[0] is not None:
                return active_staging[0].add(replay, auxiliary=auxiliary)
            value, terms = replay()
            if auxiliary is not None:
                terms = [sum(terms) + auxiliary()]
            boundary_terms.extend(terms)
            return value
        return memo.write(name, depth, compute)

    def digest_payload(record, part, leaves):
        """The digest of a listing value by the operator's plan (digest.py), every write differentiable."""
        crisp = crisp_messages(record["messages"], texts, handover_notes(record))
        opening = next((m["content"] for m in crisp if m["role"] == "user" and isinstance(m["content"], str)), "")
        found = INSTRUCTIONS.search(opening)
        system = [{"type": "neuralese", "id": leaf_ids["prompt:digest"]}]
        written = {}

        def site_write(messages):
            payload = write(messages, None, DIGEST_PREFIX, {**leaves, **written}, source=part.get("preview"))
            block = placeholder(f"{part['name']}#{len(written)}")
            written[block] = payload
            return block

        block, _ = write_digest(site_write, system, part["holder"], part["value_type"], part["source"],
                                found.group(1) if found else "", engine.tokenizer, args.digest_window)
        return written[block]

    def written_values(record, leaves, depth=0, visiting=(), memo=None):
        """Blocks written for this record this step: handoffs it reads or shows (notes, child results), digests in its
        listing. Inside a producer (depth > 0) its own target's value is not one of them. Returns (name → placeholder
        ID, placeholder ID → payload)."""
        names, payloads = {}, {}
        if memo is None:
            memo = ProducerMemo(share_producers)
        if args.handover == "written":
            own = target_write(record) if depth else None
            chosen = [name for name in sorted(reads(record) | set(handover_notes(record)))
                      if name in producers and name != own and name not in visiting]
            if args.max_writes and len(chosen) > args.max_writes:
                chosen = sorted(write_choice.sample(chosen, args.max_writes))
            for name in chosen:
                names[name] = placeholder(name)
                payloads[names[name]] = note_payload(name, leaves, depth + 1, visiting, memo)
        if args.digest == "written":
            for message in record["messages"]:
                for part in message.get("content") if isinstance(message.get("content"), list) else []:
                    if part["type"] == "digest":
                        names[part["name"]] = placeholder(part["name"])
                        payloads[names[part["name"]]] = digest_payload(record, part, leaves)
        return names, payloads

    def soft_messages(record, names):
        return render(record["messages"], lambda name: {"type": "neuralese", "id": leaf_ids[name]}, handover_notes(record),
                      names, names)

    def target_of(record, names):
        return render([record["target"]], lambda name: {"type": "text", "text": texts[name]}, handover_notes(record), names)[0]

    def prompt_tokens(record):
        crisp = crisp_messages(record["messages"], texts, handover_notes(record))
        rendered = render_messages(crisp, record.get("tools"), engine._template, engine.specials)
        return len(session._items(rendered.segments, rendered.blocks, rendered.escape_nonce))

    train, held, skipped = [], [], {"long": 0, "no-target": 0}
    with open(args.records) as stream:
        for line in stream:
            if len(train) >= args.train and len(held) >= args.eval:
                break
            record = json.loads(line)
            if not record.get("target"):
                skipped["no-target"] += 1
                continue
            if args.only_handover and not (reads(record) or handover_notes(record)):
                continue
            bucket = held if record.get("split") == "test" else train
            if len(bucket) >= (args.eval if bucket is held else args.train):
                continue
            if prompt_tokens(record) > args.max_tokens:
                skipped["long"] += 1
                continue
            bucket.append(record)
    # Soft parameters for the names the selected records use (a corpus has thousands of instructions texts).
    used_names = {part["name"] for record in train + held + list(producers.values()) for message in record["messages"]
                  if isinstance(message.get("content"), list) for part in message["content"] if part["type"] == "soft"}
    if args.digest == "written":
        used_names.add("prompt:digest")
    from .trajectory_state import soft_initialization, resumed_initial_rows
    warm_rows = soft_initialization(args.soft_init, texts, backbone.config.hidden_size, profile=heads.profile) if args.soft_init and resumed is None else {}
    saved_rows = resumed_initial_rows(resumed, used_names, backbone.config.hidden_size)
    from_previous = []
    for name in sorted(used_names):
        text = texts[name]
        piece = name.removeprefix("prompt:")
        if resumed is not None:
            # Original rows preserve block IDs; trained values and optimizer
            # state are restored below. Re-encoding here only wastes compute.
            rows = saved_rows[name]
        elif name in warm_rows:
            rows = warm_rows[name]
            from_previous.append(name)
        elif bank and name.startswith("prompt:") and piece in bank.rows:
            rows = bank.rows[piece]
            from_bank.append(name)
        else:
            rows = encode_text(engine, text).payload
        block = engine.store.put(make_block(rows, engine.dialect, type="Neuralese<SystemPrompt>",
                                            producer={"kind": "bank" if name in from_bank else "text-encode", "text": text}))
        params[name] = torch.nn.Parameter(block.payload.clone().float().to(engine.device))
        leaf_ids[name] = block.id
    init = {name: p.detach().clone() for name, p in params.items()}
    handovers = sum(1 for r in train if handover_notes(r))
    if args.handover == "written":
        # Training records that read or show a note come first in the mix's accounting; their producers may lie
        # outside the read window, which is why every producer was indexed.
        handovers = sum(1 for r in train if reads(r) or handover_notes(r))
    print(json.dumps({"train": len(train), "heldout": len(held), "skipped": skipped, "soft_params": len(params),
                      "from_bank": len(from_bank), "from_previous_soft": len(from_previous), "records_with_handover": handovers,
                      "handover": args.handover, "note_producers": len(producers)}), flush=True)

    lora = []
    if args.rank:
        groups = inject_lora(engine.backbone, list(range(engine.backbone.num_layers)), rank=args.rank, alpha=2 * args.rank)
        lora = [p for ps in groups.values() for p in ps]
    if args.crisp_weight and not lora:
        raise ValueError('crisp SFT requires trainable policy adapters (--rank positive)')
    # The writer's own modules (feedback, content projection) learn from the readers of what they write.
    # Native final normalization is a frozen reference. At full depth the
    # causal feedback is already exact; learn payload/stop without corrupting it.
    head_params = [p for name, p in heads.named_parameters()
                   if not (not heads.read_markers and (name.startswith('feedback.final_norm.') or
                           (heads.cutoff == backbone.num_layers and name.startswith('feedback.'))))] if args.heads_lr and (args.handover == "written" or args.digest == "written") else []
    for p in head_params:
        p.requires_grad_(True)
    optimizer = trajectory_optimizer(args.optimizer, params, lora, head_params,
                                     vocab_size=backbone.embedding_weight.shape[0], lr=args.lr,
                                     lora_lr=args.lora_lr, heads_lr=args.heads_lr,
                                     embedding_ids={id(p) for m in heads.modules() if isinstance(m, torch.nn.Embedding) for p in m.parameters()})
    if resumed is not None:
        if set(params) != set(resumed['params']):
            raise ValueError('recurrence soft-parameter names changed')
        with torch.no_grad():
            for name, value in resumed['params'].items():
                params[name].copy_(value.to(params[name]))
            for name, value in resumed['lora'].items():
                q = dict(backbone.hf.named_parameters())[name]
                q.copy_(value.to(q))
        heads.load_state_dict(resumed['heads'])
        optimizer.load_state_dict(resumed['optimizer'])
        init = {k: v.to(params[k]) for k, v in resumed['init'].items()}

    def count_embedding(items, dimensions):
        return sum(1 if kind == 'tok' else dimensions[value] + 2 for kind, value in items)

    def geometry_plan(record):
        dimensions = {block: params[name].shape[0] for name, block in leaf_ids.items()}
        features, memo = [], {}
        def visit(current, depth=0, visiting=()):
            own = target_write(current) if depth else None
            chosen = [name for name in sorted(reads(current) | set(handover_notes(current)))
                      if name in producers and name != own and name not in visiting]
            if args.max_writes and len(chosen) > args.max_writes:
                chosen = sorted(write_choice.sample(chosen, args.max_writes))
            names = {}
            for name in chosen:
                names[name] = placeholder(name)
                key = name, depth + 1
                if share_producers and key in memo:
                    dimensions[names[name]] = memo[key]
                    continue
                producer = producers[name]
                child_names = visit(producer, depth + 1, visiting + (name,)) if depth + 1 < args.write_depth else {}
                cache_key = name, tuple(sorted(child_names))
                if cache_key not in geometry_cache:
                    messages = render(producer['messages'], lambda n: {'type': 'neuralese', 'id': leaf_ids[n]},
                                      handover_notes(producer), child_names, child_names)
                    rendered = render_messages(messages, producer.get('tools'), engine._template, engine.specials)
                    items = session._items(rendered.segments, rendered.blocks, rendered.escape_nonce)
                    context = count_embedding(items, dimensions) + len(engine._tokens(site_prefix(producer))) + 1
                    vectors = source_length(handover_notes(producer).get(name)) or heads.max_length
                    geometry_cache[cache_key] = context, vectors
                context, vectors = geometry_cache[cache_key]
                features.append((context, vectors))
                dimensions[names[name]] = vectors
                memo[key] = vectors
            return names
        # Planning samples the identical edges without consuming training RNG.
        state = write_choice.getstate()
        try:
            names = visit(record)
            prompt, target = session._target_items(soft_messages(record, names), record.get('tools'), target_of(record, names))
            reader_context = count_embedding(prompt, dimensions) + count_embedding(target, dimensions)
            target_tokens = sum(kind == 'tok' for kind, _ in target)
        finally:
            write_choice.setstate(state)
        writer_bytes = sum(memory_estimator.predict('writer', context, vectors,
                           geometry_bytes(context, vectors, **memory_layout)) for context, vectors in features)
        reader_raw = geometry_bytes(reader_context, 0, **memory_layout, target_tokens=target_tokens,
                                    vocab_size=backbone.embedding_weight.shape[0])
        reader_bytes = memory_estimator.predict('reader', reader_context, target_tokens, reader_raw)
        return {'writers': features, 'reader_context': reader_context, 'target_tokens': target_tokens,
                'reader_raw': reader_raw, 'tape_bytes': writer_bytes + reader_bytes}

    reader_geometry = [None]
    def observe_writer(value, retained_bytes):
        context, vectors = write_context_lengths[-1], value.shape[0]
        raw = geometry_bytes(context, vectors, **memory_layout)
        memory_estimator.observe('writer', context, vectors, raw, retained_bytes)

    def loss_of(record, leaves, soft=True, training_objective=True):
        if not soft:
            crisp = crisp_messages(record["messages"], texts, handover_notes(record))
            target = render([record["target"]], lambda name: {"type": "text", "text": texts[name]}, handover_notes(record))[0]
            return session._term({"kind": "crossEntropy", "messages": crisp, "tools": record.get("tools"), "target": target}, leaves)
        # Notes and digests are written afresh by the current writer; their payloads are leaves of this loss.
        stop_terms.clear()
        boundary_terms.clear()
        names, payloads = written_values(record, leaves)
        leaves = resolve_values({**leaves, **payloads})
        messages, target = soft_messages(record, names), target_of(record, names)
        distill = args.distill if training_objective and payloads and not target_write(record) else 0
        reader_before = torch.cuda.memory_allocated() if args.device.startswith('cuda') else 0
        loss = session.supervised_text_loss(
            {"messages": messages, "tools": record.get("tools"), "target": target}, leaves,
            teacher_messages=crisp_messages(record["messages"], texts, handover_notes(record)) if distill else None,
            distill_weight=distill)
        if reader_geometry[0] and torch.is_grad_enabled() and args.device.startswith('cuda'):
            plan = reader_geometry[0]
            memory_estimator.observe('reader', plan['reader_context'], plan['target_tokens'], plan['reader_raw'],
                                     max(0, torch.cuda.memory_allocated() - reader_before))
        if not training_objective:
            # Evaluation reports pure target CE, not CE plus distillation,
            # stop-boundary penalties or policy-gradient terms.
            boundary_terms.clear()
            stop_terms.clear()
            return loss
        if active_staging[0] is not None:
            loss = loss + active_staging[0].penalty_loss(1.)
        if boundary_terms:
            loss = loss + sum(boundary_terms) / len(boundary_terms)
            boundary_terms.clear()
        if stop_terms:
            # Stop policy (phase E's objective on real readers): reward = -(reader loss + λ·length), against a running
            # baseline; the stop head learns how long a note or digest must be for what its readers need.
            reward = -(float(loss.detach()) + args.stop_pg * sum(n for _, n in stop_terms))
            advantage = reward - (baseline["value"] if baseline["value"] is not None else reward)
            baseline["value"] = reward if baseline["value"] is None else 0.9 * baseline["value"] + 0.1 * reward
            loss = loss - advantage * sum(logp for logp, _ in stop_terms)
            stop_terms.clear()
        return loss

    def evaluate(label, leaves, soft=True):
        values = []
        with torch.no_grad():
            for record in held:
                try:
                    values.append(float(loss_of(record, leaves, soft, training_objective=False)))
                except RequestError:
                    pass
        return {"label": label, "cross_entropy": sum(values) / max(1, len(values)), "n": len(values)}

    def evaluate_written(label, leaves, records):
        """Readers of written values (notes, child results, digests): reader loss with the values written for them, and
        with values written for another reader of different values (each record's placeholders filled, in order and
        cyclically, from the next such reader's payloads). Written must beat shuffled for the values to carry content."""
        stop_terms.clear()
        boundary_terms.clear()
        sites, write_errors, reader_errors, missing_donors = [], [], [], []
        with torch.no_grad():
            for record in records:
                try:
                    names, payloads = written_values(record, leaves)
                except RequestError as error:
                    write_errors.append({'id': record['id'], 'error': str(error)})
                    continue
                if payloads:
                    sites.append((record, names, payloads))
        result = {"label": label, "n": 0, "expected_n": len(sites) + len(write_errors),
                  "write_errors": write_errors, "reader_errors": reader_errors, "missing_donors": missing_donors}
        if len(sites) < 2:
            return result
        own, shuffled = [], []
        with torch.no_grad():
            for i, (record, names, payloads) in enumerate(sites):
                others = [sites[(i + k) % len(sites)] for k in range(1, len(sites))]
                donor = next((list(p.values()) for _, n, p in others if set(n) != set(names)), None)
                if donor is None:
                    missing_donors.append(record['id'])
                    continue
                swapped = {key: donor[k % len(donor)] for k, key in enumerate(payloads)}
                def reader(values):
                    return float(session._term({"kind": "crossEntropy", "messages": soft_messages(record, names),
                                                "tools": record.get("tools"), "target": target_of(record, names)},
                                               {**leaves, **values}))
                try:
                    a, b = reader(payloads), reader(swapped)
                except RequestError as error:
                    reader_errors.append({'id': record['id'], 'error': str(error)})
                    continue
                own.append(a)
                shuffled.append(b)
        stop_terms.clear()
        boundary_terms.clear()
        if own:
            result.update({"written": sum(own) / len(own), "shuffled": sum(shuffled) / len(shuffled), "n": len(own),
                           "written_better": sum(a < b for a, b in zip(own, shuffled)) / len(own)})
        return result

    leaves = {leaf_ids[name]: p for name, p in params.items()}
    report = resumed['initial_report'] if resumed is not None else {"crisp": evaluate("crisp", {}, soft=False), "soft-init": evaluate("soft-init", leaves)}
    if resumed is None and (args.handover == "written" or args.digest == "written"):
        probe = train[:args.eval]
        report["written-init"] = evaluate_written("written-init", leaves, held)
        report["written-init-train"] = evaluate_written("written-init-train", leaves, probe)
    print(json.dumps(report), flush=True)
    probe = train[:args.eval]
    log = open(out / "train.jsonl", "a" if resumed is not None else "w")
    started, cursor, errors, used = time.time(), 0, 0, set()
    start_step = 0
    if resumed is not None:
        start_step, cursor, errors, used = resumed['step'], resumed['cursor'], resumed['errors'], set(resumed['used'])
        baseline.update(resumed['baseline'])
        random.setstate(resumed['python_rng'])
        write_choice.setstate(resumed['write_rng'])
        stop_generator.set_state(resumed['stop_rng'])
        torch.set_rng_state(resumed['torch_rng'])
        if args.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(resumed['cuda_rng'])
    graph_routes = dict(resumed.get('graph_routes', {})) if resumed else {}
    trainables = list(params.values()) + lora + head_params
    stop_requested = [False]
    previous_handlers = {sig: signal.signal(sig, lambda *_: stop_requested.__setitem__(0, True))
                         for sig in (signal.SIGTERM, signal.SIGINT)}
    best_evaluation = resumed.get('best_evaluation') if resumed else None
    def save_training_state(step, destination=None):
        atomic_checkpoint(destination or checkpoint_path, {
            'schema': 'natlang.neuralese_recurrence_checkpoint/1', 'identity': identity, 'graph_routes': graph_routes, 'memory_estimator': memory_estimator.state_dict(),
            'step': step, 'cursor': cursor, 'errors': errors, 'used': sorted(used), 'best_evaluation': best_evaluation,
            'params': {k: v.detach().cpu() for k, v in params.items()}, 'texts': texts,
            'control_rows': backbone.control_rows.detach().cpu(),
            'port_config': {'cutoff': heads.cutoff, 'max_length': heads.max_length, **heads.port_config()},
            'heads': heads.state_dict(), 'lora': lora_state(backbone), 'optimizer': optimizer.state_dict(),
            'init': {k: v.detach().cpu() for k, v in init.items()}, 'initial_report': report,
            'baseline': baseline, 'python_rng': random.getstate(), 'write_rng': write_choice.getstate(),
            'stop_rng': stop_generator.get_state(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else []})
    save_training_state(start_step)
    with torch.enable_grad():
        for step in range(start_step, args.steps):
            step_started = time.time()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            crisp_losses = []
            released_graph_bytes = 0
            offload_stats = {'offloaded_bytes': 0}
            staged_nodes, replay_error = 0, 0.0
            step_peak_bytes = 0
            step_record_ids = []
            for _ in range(args.batch):
                if args.device.startswith("cuda"):
                    torch.cuda.reset_peak_memory_stats()
                record = train[cursor % len(train)]
                step_record_ids.append(record['id'])
                cursor += 1
                mode = args.backward_policy
                plan = geometry_plan(record) if args.backward_policy == 'auto' else None
                reader_geometry[0] = plan
                if mode == 'auto':
                    baseline_bytes = torch.cuda.memory_allocated() if args.device.startswith('cuda') else 0
                    raw_prediction = baseline_bytes + plan['tape_bytes']
                    predicted = memory_estimator.adjust_joint(plan, raw_prediction)
                    mode = 'staged' if predicted > graph_budget or graph_routes.get(record['id'], 0) >= graph_budget else 'joint'
                    with (out / 'memory-routing.jsonl').open('a') as routing_log:
                        routing_log.write(json.dumps({'step': step, 'record_id': record['id'], 'mode': mode,
                            'estimated_gib': predicted / 2**30, 'graph_budget_gib': graph_budget / 2**30,
                            'write_sites': len(plan['writers'])}) + '\n')
                rng_before = (random.getstate(), write_choice.getstate(), stop_generator.get_state(),
                              torch.get_rng_state(), torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
                              dict(baseline), len(lengths), len(write_context_lengths))
                def attempt_joint():
                    # autograd.grad avoids partial parameter .grad mutations on
                    # an aborted attempt, including accumulated earlier chains.
                    try:
                        with graph_memory_budget(graph_budget):
                            objective = loss_of(record, leaves) / args.batch
                            gradients = torch.autograd.grad(objective, trainables, allow_unused=True)
                        peak = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
                        return float(objective.detach()), gradients, None, peak
                    except (GraphBudgetExceeded, torch.OutOfMemoryError) as failure:
                        peak = torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else 0
                        return None, None, type(failure).__name__, peak
                try:
                    if args.backward_policy == 'auto' and mode == 'joint':
                        value, gradients, failure, joint_peak = attempt_joint()
                        memory_estimator.observe_joint(plan, raw_prediction, joint_peak, failed=bool(failure))
                        if failure:
                            graph_routes[record['id']] = int(graph_budget) + 1
                            random.setstate(rng_before[0])
                            write_choice.setstate(rng_before[1])
                            stop_generator.set_state(rng_before[2])
                            torch.set_rng_state(rng_before[3])
                            if rng_before[4]:
                                torch.cuda.set_rng_state_all(rng_before[4])
                            baseline.clear()
                            baseline.update(rng_before[5])
                            del lengths[rng_before[6]:]
                            del write_context_lengths[rng_before[7]:]
                            boundary_terms.clear()
                            stop_terms.clear()
                            gc.collect()
                            if args.device.startswith('cuda'):
                                torch.cuda.empty_cache()
                            mode = 'staged'
                            print(json.dumps({'status': 'stage_for_budget', 'record_id': record['id'],
                                              'joint_failure': failure, 'graph_budget_gib': graph_budget / 2**30}), flush=True)
                        else:
                            for param, gradient in zip(trainables, gradients):
                                if gradient is not None:
                                    if param.grad is None:
                                        param.grad = gradient
                                    else:
                                        param.grad.add_(gradient)
                            del gradients
                            losses.append(value * args.batch)
                    if mode == 'staged' or args.backward_policy == 'joint':
                        active_staging[0] = StagedWrites(observe=observe_writer, measure=torch.cuda.memory_allocated if args.device.startswith('cuda') else None) if mode == 'staged' else None
                        from .memory import offload_attention_tensors
                        persistent = list(backbone.parameters()) + list(backbone.buffers()) + list(heads.parameters()) + list(heads.buffers()) + list(params.values())
                        with offload_attention_tensors(int(args.activation_offload_gb * 2**30), activations=True,
                                                      persistent_tensors=persistent) as offload_stats:
                            loss = loss_of(record, leaves) / args.batch
                        loss.backward()
                        losses.append(float(loss.detach()) * args.batch)
                        del loss
                        gc.collect()
                        if active_staging[0] is not None:
                            active_staging[0].backward(penalty_weight=1., scale=1 / args.batch)
                            staged_nodes += len(active_staging[0].nodes)
                            replay_error = max(replay_error, active_staging[0].replay_max_abs_error)
                            active_staging[0].clear()
                            active_staging[0] = None
                    # The policy learns from the exact ordinary runtime too. Its graph
                    # is built after the recurrent graph has been freed; all gradients
                    # accumulate before one optimizer step, never doubling live tapes.
                    if args.crisp_weight:
                        objective = args.crisp_weight * loss_of(record, {}, soft=False) / args.batch
                        gradients = torch.autograd.grad(objective, trainables, allow_unused=True)
                        for param, gradient in zip(trainables, gradients):
                            if gradient is not None:
                                if param.grad is None:
                                    param.grad = gradient
                                else:
                                    param.grad.add_(gradient)
                        crisp_losses.append(float(objective.detach()) * args.batch)
                        del objective, gradients
                    gc.collect()
                    case_peak = torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0
                    step_peak_bytes = max(step_peak_bytes, case_peak)
                    if args.backward_policy == 'auto':
                        with (out / 'memory-routing.jsonl').open('a') as routing_log:
                            routing_log.write(json.dumps({'event': 'result', 'step': step, 'record_id': record['id'],
                                'mode': mode, 'observed_peak_gib': case_peak / 2**30,
                                'estimated_gib': predicted / 2**30}) + '\n')
                except RequestError:
                    if active_staging[0] is not None:
                        active_staging[0].clear()
                        active_staging[0] = None
                    errors += 1
                    continue
                except torch.OutOfMemoryError:
                    failure = {'status': 'training_out_of_memory', 'step': step, 'record_id': record['id'],
                               'curriculum': args.write_curriculum, 'batch': args.batch, 'backward_mode': mode}
                    (out / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
                    print(json.dumps(failure), flush=True)
                    raise
                used.update(part["name"] for m in record["messages"] if isinstance(m.get("content"), list)
                            for part in m["content"] if part["type"] == "soft")
            reader_geometry[0] = None
            # The writer's gradient from its readers: zero would mean written values do not train the writer.
            writer_grad = float(gradient_norm(head_params)) if head_params else None
            clip_finite_gradients(trainables, 1.0)
            optimizer.step()
            entry = {"step": step, "reader_record_ids": step_record_ids, "loss": sum(losses) / max(1, len(losses)), "seconds": round(time.time() - started),
                     "errors": errors, "backward_mode": mode, "staged_nodes": staged_nodes,
                     "replay_max_abs_error": replay_error, "crisp_sft_loss": sum(crisp_losses) / max(1, len(crisp_losses)), "step_seconds": round(time.time() - step_started, 3), **({"writer_grad_norm": writer_grad} if head_params else {}),
                     **({"write_lengths": lengths[-8:]} if lengths else {})}
            if args.device.startswith("cuda"):
                entry["peak_gb"] = round(max(step_peak_bytes, torch.cuda.max_memory_allocated()) / 2**30, 2)
                entry['released_graph_gib'] = round(released_graph_bytes / 2**30, 3)
                entry['largest_write_context_tokens'] = max(write_context_lengths, default=0)
                entry['activation_offloaded_gib'] = round(offload_stats['offloaded_bytes'] / 2**30, 3)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if step % 10 == 0 or step == args.steps - 1:
                print(json.dumps(entry), flush=True)
            if args.eval_every and (step + 1) % args.eval_every == 0 and not stop_requested[0]:
                from .trajectory_state import evaluation_state
                with evaluation_state(write_choice, stop_generator, baseline):
                    evaluation = {'step': step + 1, 'soft': evaluate('periodic-soft', leaves)}
                    if args.crisp_weight:
                        evaluation['crisp'] = evaluate('periodic-crisp', {}, soft=False)
                    if args.handover == 'written' or args.digest == 'written':
                        evaluation['written'] = evaluate_written('periodic-written', leaves, held)
                with (out / 'eval.jsonl').open('a') as evaluation_log:
                    evaluation_log.write(json.dumps(evaluation) + '\n')
                print(json.dumps({'evaluation': evaluation}), flush=True)
                written = evaluation.get('written', {})
                score = written.get('written')
                from .trajectory_state import paired_probe_complete
                if (score is not None and paired_probe_complete(written)
                        and written.get('shuffled', score) > score
                        and (best_evaluation is None or score < best_evaluation['written'])):
                    best_evaluation = {'step': step + 1, **written, 'semantic_channel_qualified': False,
                                       'selection_scope': 'candidate by complete paired reader CE; separate semantic/stopping eval required'}
                    save_training_state(step + 1, out / 'best-checkpoint.pt')
                    (out / 'best-evaluation.json').write_text(json.dumps(best_evaluation, indent=2) + '\n')
            if (step + 1) % args.checkpoint_every == 0 or stop_requested[0] or step + 1 == args.steps:
                save_training_state(step + 1)
            if stop_requested[0]:
                break
    for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)
    log.close()
    if stop_requested[0]:
        print(json.dumps({'status': 'checkpointed_on_signal', 'checkpoint': str(checkpoint_path)}), flush=True)
        return 0
    report["soft-trained"] = evaluate("soft-trained", leaves)
    if args.handover == "written" or args.digest == "written":
        report["written-trained"] = evaluate_written("written-trained", leaves, held)
        report["written-trained-train"] = evaluate_written("written-trained-train", leaves, probe)
    if lengths:
        report["writes"] = {"count": len(lengths), "mean_length": sum(lengths) / len(lengths), "max_length": max(lengths)}
    if head_params:
        from .adapters import adapter_layers
        source_metadata = torch.load(args.heads, map_location='cpu', weights_only=False, mmap=True) if args.heads else {}
        ranks = {int(p.shape[0]) for name, p in backbone.hf.named_parameters() if '.lora_A.' in name}
        if len(ranks) > 1:
            raise ValueError('cannot export mixed-rank adapters in one port checkpoint')
        atomic_checkpoint(out / 'heads.pt', {'heads': heads.state_dict(),
            'port_config': {**source_metadata.get('port_config', {}), 'cutoff': heads.cutoff,
                            'max_length': heads.max_length, **heads.port_config()},
            'control_rows': backbone.control_rows.detach().cpu(), 'lora': lora_state(backbone),
            'lora_layers': adapter_layers(backbone), 'lora_rank': next(iter(ranks), 0),
            'backbone': source_metadata.get('backbone') or {'base': args.base},
            'training_identity': identity})
    # Every soft parameter's movement: also those only producers' contexts hold, which move by their readers' losses.
    moved = {name: float((params[name].detach() - init[name]).norm() / init[name].norm().clamp_min(1e-9)) for name in params}
    report["relative_change"] = moved
    torch.save({"params": {k: v.detach().cpu() for k, v in params.items()}, "texts": texts,
                'port_profile': heads.profile}, out / "soft-params.pt")
    if lora:
        torch.save({"lora": lora_state(engine.backbone), "rank": args.rank}, out / "adapter.pt")
    if bank:
        current = {name.removeprefix("prompt:"): params[name] for name in used if name.removeprefix("prompt:") in bank.texts}
        save_bank(out / "system-prompts.nz", bank, current, {"kind": "system-prompt-bank", "init": args.bank,
                  "trained_by": "natlang_neuralese.train.trajectories", "records": args.records, "steps": args.steps})
    (out / "summary.json").write_text(json.dumps({"options": vars(args), **report}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "relative_change"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
