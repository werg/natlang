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

from .memory import cuda_allocated_bytes

from ..digest import PREFIX as DIGEST_PREFIX, digest_note, write_digest

INSTRUCTIONS = re.compile(r"Instructions:\n([\s\S]*?)\n\n(?:In eval|Eval also|$)")


def crisp_messages(messages: list[dict], texts: dict[str, str], notes: dict[str, str], *,
                   neuralese_bodies: dict[str, str] | None = None) -> list[dict]:
    """Messages with every soft part as its text and every handover as its note: the crisp rendering."""
    return render(messages, lambda name: {"type": "text", "text": texts[name]}, notes,
                  neuralese_bodies=neuralese_bodies)


def render(messages: list[dict], soft_part, notes: dict[str, str], blocks: dict[str, str] | None = None,
           digests: dict[str, str] | None = None,
           neuralese_bodies: dict[str, str] | None = None) -> list[dict]:
    """Converted messages → engine messages: `soft` parts via `soft_part(name)`; handover reads and writes as the
    written block where `blocks` has one (name → block ID), else as the crisp note; parts merged into text where no
    block remains."""
    blocks, digests, neuralese_bodies = blocks or {}, digests or {}, neuralese_bodies or {}
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
                elif part["type"] == "neuralese":
                    # Ordinary text training can expand only a separately
                    # attested creation body. Other native blocks stay opaque.
                    body = neuralese_bodies.get(part.get("id"))
                    if body is None:
                        raise ValueError("unattested neuralese message body")
                    parts.append({"type": "text", "text": body})
                else:
                    parts.append({"type": "text", "text": part["text"]})
            message["content"] = ("".join(p["text"] for p in parts) if all(p["type"] == "text" for p in parts) else parts)
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                args = call["function"]["arguments"]
                if 'neuralese_code' in call:
                    from .inline_instructions import render_inline_instruction_arguments
                    rendered_args=render_inline_instruction_arguments(args,call['neuralese_code'],blocks)
                    if rendered_args is None:raise ValueError('invalid inline instruction code sidecar')
                    call={k:v for k,v in call.items() if k!='neuralese_code'}
                    call={**call,'function':{**call['function'],'arguments':rendered_args}}
                    calls.append(call)
                    continue
                if '"$write"' in args:
                    def crisp_value(value):
                        """Expand write leaves recursively, retaining opaque blocks at their exact value path."""
                        if isinstance(value, dict) and "$write" in value:
                            site = value["$write"]
                            name = site["name"]
                            if name in blocks:
                                return [{"type": "neuralese", "id": blocks[name],
                                         "value_type": "string" if site.get("type", "Neuralese<string>") == "Neuralese<string>" else "unknown"}]
                            source = site["source"]
                            return json.loads(source) if site.get("type") == "Neuralese<unknown>" else source
                        if isinstance(value, dict):
                            return {key: crisp_value(item) for key, item in value.items()}
                        if isinstance(value, list):
                            return [crisp_value(item) for item in value]
                        return value

                    value = crisp_value(json.loads(args))
                    arguments = json.dumps(value, ensure_ascii=False)
                    call = {**call, "function": {**call["function"], "arguments": arguments}}
                calls.append(call)
            message["tool_calls"] = calls
        out.append(message)
    return out


def reads(record: dict) -> set[str]:
    return {part["name"] for m in record["messages"] if isinstance(m.get("content"), list)
            for part in m["content"] if part["type"] == "read"}


def write_sites(record: dict) -> list[tuple[str, dict, str, str]]:
    """Every actual target writer, including multiple attested inline bodies in one eval."""
    sites=[]
    for call in (record.get('target') or {}).get('tool_calls') or []:
        arguments=json.loads(call['function']['arguments'])
        before={}
        for key,value in arguments.items():
            if isinstance(value,dict) and '$write' in value:
                sites.append((call['function']['name'],dict(before),key,value['$write']['name']))
            before[key]=value
        if 'neuralese_code' in call:
            from .inline_instructions import validate_inline_instruction_code
            checked=validate_inline_instruction_code(call['function']['arguments'],call['neuralese_code'])
            if not checked.valid:raise ValueError('invalid inline instruction code sidecar: '+str(checked.reason))
            before={}
            for key,value in arguments.items():
                if key=='code':break
                before[key]=value
            for write in checked.value.writes:
                sites.append((call['function']['name'],dict(before),'code',write.name))
    return sites


def write_site(record: dict) -> tuple[str, dict, str, str] | None:
    selected=record.get('_active_write_name')
    return next((site for site in write_sites(record) if selected is None or site[3]==selected),None)


def inline_write_prefix(record: dict, name: str) -> str | None:
    from .inline_instructions import validate_inline_instruction_code
    for call in (record.get('target') or {}).get('tool_calls') or []:
        if 'neuralese_code' not in call:continue
        checked=validate_inline_instruction_code(call['function']['arguments'],call['neuralese_code'])
        if not checked.valid:raise ValueError('invalid inline instruction code sidecar: '+str(checked.reason))
        pair=checked.value.prefix_and_body(name)
        if pair is not None:return pair[0]
    return None


def write_value_type(record: dict) -> str:
    site=write_site(record)
    if site is None:return 'string'
    name=site[3]
    if inline_write_prefix(record,name) is not None:return 'string'
    for call in (record.get('target') or {}).get('tool_calls') or []:
        for value in json.loads(call['function']['arguments']).values():
            if isinstance(value,dict) and '$write' in value and value['$write']['name']==name:
                return 'string' if value['$write'].get('type','Neuralese<string>')=='Neuralese<string>' else 'unknown'
    raise ValueError('selected writer contract is missing')


def target_writes(record: dict) -> set[str]:
    return {site[3] for site in write_sites(record)}


def target_write(record: dict) -> str | None:
    site=write_site(record)
    return site[3] if site else None


def handover_notes(record: dict) -> dict[str, str]:
    """Note texts by name: from the record's compaction calls and from its reads (which carry their note)."""
    notes = {part["name"]: part["source"] for m in record["messages"] if isinstance(m.get("content"), list)
             for part in m["content"] if part["type"] == "read" and "source" in part}
    for message in record["messages"] + ([record["target"]] if record.get("target") else []):
        for call in message.get("tool_calls") or []:
            if 'neuralese_code' in call:
                from .inline_instructions import validate_inline_instruction_code
                checked=validate_inline_instruction_code(call['function']['arguments'],call['neuralese_code'])
                if not checked.valid:raise ValueError('invalid inline instruction code sidecar: '+str(checked.reason))
                notes.update({write.name:write.source for write in checked.value.writes})
            if '"$write"' in call["function"]["arguments"]:
                for value in json.loads(call["function"]["arguments"]).values():
                    if isinstance(value, dict) and "$write" in value:
                        notes[value["$write"]["name"]] = value["$write"]["source"]
    return notes


def native_writer_prefix(record, apply_template, *, value_type=None):
    """Actual template reply through the chosen argument and any preceding eval code."""
    from ..serve.chat import write_reply,write_value_text
    call,before,argument,name=write_site(record)
    value_type=value_type or write_value_type(record)
    prefix=write_reply(apply_template,call,before,argument,value_type)[0]
    code_prefix=inline_write_prefix(record,name)
    if code_prefix is not None:
        prefix+=write_value_text(apply_template,call,before,argument,code_prefix,value_type)
    return prefix


def producer_text_target(record, texts, names):
    """Gold ordinary reply at a producer, preserving its incoming soft values.

    Its own output must stay gold text here, so body-token supervision is not
    replaced by the generated opaque payload that the reader objective consumes.
    """
    own = target_writes(record)
    ancestors = {name: block for name, block in names.items() if name not in own}
    return render([record['target']], lambda name: {'type': 'text', 'text': texts[name]},
                  handover_notes(record), ancestors)[0]




def _progress(records, label: str):
    """Yield records, logging each one's wall time to stderr: evaluation passes run before the first update and
    otherwise print nothing until done, so a slow backbone looks like a hang."""
    import sys
    import time
    start = time.time()
    for index, record in enumerate(records):
        began = time.time()
        yield record
        print(json.dumps({"event": "eval_progress", "label": label, "record": index + 1, "of": len(records),
                          "seconds": round(time.time() - began, 2), "elapsed": round(time.time() - start, 1)}),
              file=sys.stderr, flush=True)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", required=True)
    parser.add_argument("--pieces", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--crisp-weight", type=float, default=0.0, help="additional ordinary-text SFT, backward separately before the same optimizer step; preserves interpreter policy alongside soft-return learning")
    parser.add_argument("--writer-text-weight", type=float, default=None, help="teacher-forced gold producer reply under its actual soft/ancestor context; additional local writer objective")
    parser.add_argument("--stop-supervision", choices=["generated-length", "gold-native-boundary"], default="generated-length", help="teach stop on coherent gold value states with balanced terminal/continue loss")
    parser.add_argument("--writer-supervision", choices=["full-reply", "native-value"], default="native-value", help="teacher-force the gold body under the exact forced writer prefix; full-reply reproduces earlier supervision")
    parser.add_argument("--writer-length-policy", choices=["source-text", "native-value"], default="native-value", help="supervised producer length uses its exact native template value, not source JSON")
    parser.add_argument("--content-transport", choices=["learned-residual", "raw-identity", "top-state"], default="learned-residual", help="explicit raw identity warm-up or learned content residual")
    parser.add_argument("--content-residual-initialization", choices=["preserve", "fresh-zero"], default="preserve",
                        help="explicit raw-to-learned transition: zero previously bypassed residual and only its optimizer slots")
    parser.add_argument("--curriculum-change", action="append", default=[], choices=["tokens_per_vector", "writer_text_weight", "write_depth", "write_curriculum", "max_writes", "max_write_vectors", "content_transport", "content_residual_initialization", "writer_length_policy", "writer_supervision", "stop_supervision", "steps", "sketch_gradient"], help="explicitly permit named curriculum changes at --continue-from while preserving optimizer/RNG and fixed data")
    parser.add_argument('--max-write-vectors', type=int, default=None,
                        help='explicit port payload bound, distinct from prompt context; constant-stop capacity can extend without changing weights/moments')
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
    parser.add_argument("--backbone-lr", type=float, default=3e-5,
                        help="learning rate for native full-layer backbone training")
    parser.add_argument("--backbone-training", choices=["auto", "full", "lora", "qat"], default="auto",
                        help="auto selects native full-layer training or Maple QAT; lora is an explicit diagnostic policy")
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
    parser.add_argument("--sketch-gradient", choices=["unroll", "one_step", "local_stage"], default="unroll")
    parser.add_argument("--local-stage-batch-size", type=int, default=1, help="isolated sketch stages per tensor batch;1 is sequential reference; explicit memory/performance control")
    parser.add_argument("--sketch-target-backbone-scale", type=float, default=0.05, help="auxiliary sketch-target input gradient multiplier; projection receives full gradient")
    parser.add_argument("--sketch-target-weight", type=float, default=0.)
    parser.add_argument("--train-control-rows", action=argparse.BooleanOptionalAction, help="train/save/restore LM control rows for close-token stopping")
    parser.add_argument("--stop-weight", type=float, default=1.0, help="weight of the stop-boundary loss on source-sized writes")
    parser.add_argument("--heads-lr", type=float, default=1e-4, help="the writer's port heads, when notes or digests are written")
    parser.add_argument("--detach-write-context", action="store_true",
                        help="no gradient into the write sites' context (saves memory; soft prompts there then do not learn from writing)")
    parser.add_argument("--distill", type=float, default=1.0, help="weight of the self-distillation term on written notes")
    parser.add_argument("--context-weight", type=float, default=1.0, help="CE weight of each record's new prompt text (instructions, inputs, tool results since the last assistant reply); the whole trajectory is a target, not only replies (owner, 2026-10-07)")
    parser.add_argument("--feedback-weight", type=float, default=0.25, help="relative weight, inside --context-weight, of tool results and other mechanical feedback (non-assistant turns after the first reply)")
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
    parser.add_argument('--checkpoint-layers', action=argparse.BooleanOptionalAction,
                        help='recompute layer activations during backward to reduce memory; preserves full recurrence gradients')
    parser.add_argument('--checkpoint-elide-rng', action='store_true',
                        help='skip RNG snapshot/restore inside verified deterministic native layer checkpoints')
    parser.add_argument('--producer-batch-size', type=int, default=1,
                        help='maximum tensor batch of independent staged raw/native producer calls; memory admission can choose fewer')
    parser.add_argument('--joint-producer-batching', action='store_true',
                        help='also batch independent raw/native writers while retaining the complete joint graph')
    parser.add_argument('--producer-batch-memory-gb', type=float, default=0,
                        help='optional budget for one released staged frontier, separate from the complete joint graph budget')
    parser.add_argument('--checkpoint-attention-only', action='store_true',
                        help='opt-in resource policy: checkpoint attention, retain convolution activations; requires --checkpoint-layers and more graph memory')
    parser.add_argument('--staged-checkpoint-attention-only', action='store_true',
                        help='opt-in: retain convolution activations only for staged writer replays whose geometry fits the graph budget; preserve full reader/joint checkpoints')
    parser.add_argument('--ffn-chunk-tokens', type=int, default=0,
                        help='token-local FFN chunks reduce transient allocations without context truncation')
    parser.add_argument('--token-cache-mib', type=int, default=0,
                        help='bounded CPU token-ID cache for this fixed tokenizer; 0 disables it')
    parser.add_argument('--optimizer', choices=['adamw', 'muon'], default='adamw')
    parser.add_argument('--checkpoint-every', type=int, default=25)
    parser.add_argument('--eval-every', type=int, default=0, help='periodic held-out soft and written-vs-shuffled probes; 0: initial/final only')
    from .sketch_defaults import apply_sketch_defaults
    apply_sketch_defaults(parser)
    parser.add_argument("--inspect-training-config", action="store_true", help="print effective defaults and overrides without loading models or starting training")
    args = parser.parse_args(argv)
    if args.inspect_training_config:
        print(json.dumps(vars(args), sort_keys=True, indent=2))
        return
    if args.token_cache_mib < 0:
        raise ValueError('negative token cache budget')
    if args.local_stage_batch_size < 1 or (args.local_stage_batch_size != 1 and args.sketch_gradient != 'local_stage'):
        raise ValueError('local stage batches require local_stage and a positive size')
    if args.producer_batch_size < 1:
        raise ValueError('producer batch size must be positive')
    if args.joint_producer_batching and args.producer_batch_size < 2:
        raise ValueError('joint producer batching requires producer batch size >=2')
    if not math.isfinite(args.producer_batch_memory_gb) or args.producer_batch_memory_gb < 0:
        raise ValueError('invalid producer batch memory budget')
    if args.producer_batch_size > 1 and (args.tokens_per_vector != 1 or args.content_transport != 'raw-identity'
            or args.writer_supervision != 'native-value' or args.stop_supervision != 'gold-native-boundary'
            or args.stop_pg or args.max_writes or args.write_curriculum != 'joint'):
        raise ValueError('producer batching requires deterministic raw/native gold-boundary full-DAG supervision')
    from .staging import StagedWrites, resolve_values, GraphBudgetExceeded, graph_memory_budget
    from .recurrence import curriculum_max_writes
    args.max_writes = curriculum_max_writes(args.write_curriculum, args.max_writes)
    if args.batch is None:
        args.batch = 1 if args.write_curriculum == 'sampled-chain' else 4
    if args.steps < 1 or args.batch < 1 or args.checkpoint_every < 1 or args.write_depth < 1 or args.activation_offload_gb < 0 or args.ffn_chunk_tokens < 0 or args.eval_every < 0:
        raise ValueError('invalid recurrence training controls')
    if (args.checkpoint_attention_only or args.staged_checkpoint_attention_only) and not args.checkpoint_layers:
        raise ValueError('attention-only checkpointing requires --checkpoint-layers')
    if args.checkpoint_attention_only and args.staged_checkpoint_attention_only:
        raise ValueError('choose global or staged-writer attention-only checkpointing')
    if args.max_write_vectors is not None and args.max_write_vectors < 1:
        raise ValueError('invalid write capacity')
    if not math.isfinite(args.tokens_per_vector) or args.tokens_per_vector < 0:
        raise ValueError('invalid tokens per vector')
    if args.curriculum_change and not args.continue_from:
        raise ValueError('curriculum changes require explicit continuation checkpoint')
    if args.stop_supervision == 'gold-native-boundary' and (args.writer_supervision != 'native-value' or args.tokens_per_vector != 1):
        raise ValueError('gold native stop supervision requires native value supervision and one token per vector')
    if args.writer_supervision == 'native-value' and args.writer_length_policy != 'native-value':
        raise ValueError('native-value writer supervision requires native-value length policy')
    if args.writer_text_weight is not None and (not math.isfinite(args.writer_text_weight) or args.writer_text_weight < 0):
        raise ValueError('invalid producer text supervision weight')
    if not math.isfinite(args.crisp_weight) or args.crisp_weight < 0:
        raise ValueError('invalid crisp SFT weight')
    if args.graph_memory_gb < 0 or args.graph_headroom_gb <= 0:
        raise ValueError('invalid graph memory budget')
    if args.backward_policy != 'joint' and (args.stop_pg or args.digest == 'written'):
        raise ValueError('staging currently requires deterministic handoffs and preview digests')
    if args.device.startswith('cuda'):
        free, total = torch.cuda.mem_get_info()
        args.memory_gb = args.memory_gb or float(os.environ.get('NATLANG_CUDA_MEMORY_GB') or free / 2**30 * .9)
        envelope = min(args.memory_gb, total / 2**30)
        graph_budget = (args.graph_memory_gb or envelope - args.graph_headroom_gb) * 2**30
        if graph_budget <= 0 or graph_budget >= envelope * 2**30:
            raise ValueError('graph budget must leave backward headroom within the CUDA envelope')
        if args.producer_batch_memory_gb and args.producer_batch_memory_gb > envelope - args.graph_headroom_gb:
            raise ValueError('producer batch budget exceeds the envelope with backward headroom')
    else:
        graph_budget = 0
        args.memory_gb = args.memory_gb or 8
    batch_graph_budget = args.producer_batch_memory_gb * 2**30 or graph_budget

    from ..prompt_bank import load_bank, save_bank
    from ..serve import load_engine
    from ..serve.chat import RequestError, call_reply, write_reply, render_messages
    from ..serve.grad import GradSession, encode_text
    from .execution import Prefilled, write_generated
    from .losses import stop_boundary_loss
    from ..serve.store import make_block
    from .adapters import lora_state

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
    identity = {'options': {k: v for k, v in vars(args).items() if k not in {'inspect_training_config', 'out', 'memory_gb', 'activation_offload_gb', 'checkpoint_every', 'backward_policy', 'graph_memory_gb', 'graph_headroom_gb', 'continue_from', 'curriculum_change', 'checkpoint_attention_only', 'staged_checkpoint_attention_only', 'checkpoint_elide_rng', 'producer_batch_size', 'producer_batch_memory_gb', 'token_cache_mib', 'joint_producer_batching', 'local_stage_batch_size'} and not (k == 'writer_text_weight' and v is None)},
                'files': {str(Path(p).resolve()): digest_file(p) for p in [args.records, args.pieces, args.heads, args.bank, args.soft_init] if p},
                'code': {str(p.resolve()): digest_file(p) for p in Path(__file__).resolve().parents[1].rglob('*.py')}}
    if args.continue_from:
        identity['continuation'] = {'checkpoint_sha256': digest_file(args.continue_from),
                                    'path': str(Path(args.continue_from).resolve()),
                                    'curriculum_changes': args.curriculum_change}
    resumed = torch.load(checkpoint_path, map_location='cpu', weights_only=False) if checkpoint_path.exists() else None
    new_continuation = resumed is None and bool(args.continue_from)
    if resumed is not None:
        validate_resume(resumed, identity)
    elif args.continue_from:
        resumed = torch.load(args.continue_from, map_location='cpu', weights_only=False)
        validate_continuation(resumed, identity, allowed_changes=args.curriculum_change)
    if args.content_residual_initialization == 'fresh-zero':
        if not args.continue_from or args.content_transport != 'learned-residual':
            raise ValueError('fresh-zero residual requires an explicit learned-residual continuation')
        if new_continuation and (resumed.get('port_config', {}).get('content_transport') != 'raw-identity'
                or 'content_residual_initialization' not in args.curriculum_change):
            raise ValueError('fresh-zero transition must explicitly declare a previously bypassed raw residual')
    if args.device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_gb * 2**30 / total))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    engine = load_engine(args.base, heads_checkpoint=args.heads, device=args.device)
    if args.token_cache_mib:
        from ..serve.token_cache import TokenCache
        engine._token_cache = TokenCache(engine.tokenizer, args.token_cache_mib * 2**20)
    from ..model.capacity import set_write_capacity, source_vector_length
    capacity = args.max_write_vectors
    if capacity is None:
        capacity = (resumed or {}).get('port_config', {}).get('max_length', engine.heads.max_length)
    set_write_capacity(engine.heads, capacity)
    engine.heads.set_content_transport(args.content_transport)
    engine.max_block = capacity
    if args.stop_supervision == 'gold-native-boundary' and engine.heads.read_markers:
        raise ValueError('gold native stop supervision requires raw-token-v1')
    for p in engine.backbone.parameters():
        p.requires_grad_(False)
    if not 0 <= args.sketch_target_backbone_scale <= 1:
        raise ValueError("sketch target backbone gradient scale must be between 0 and 1")
    if args.sketch_target_weight < 0 or (args.sketch_target_weight and args.sketch_gradient not in ('one_step', 'local_stage')):
        raise ValueError('positive sketch target weight requires one_step or local_stage')
    if engine.heads.autoregressive and (args.sketch_gradient not in ('one_step', 'local_stage') or args.sketch_target_weight <= 0):
        raise ValueError('latent-sketch-v2 training requires one_step or local_stage and a positive sketch target weight')
    if engine.heads.autoregressive and args.stop_weight and not args.train_control_rows:
        raise ValueError('close-token stop training requires explicit --train-control-rows')
    session = GradSession(engine)

    texts, piece_kinds = {}, {}
    with open(args.pieces) as stream:
        for line in stream:
            piece = json.loads(line)
            texts[piece["name"]] = piece["text"]
            piece_kinds[piece["name"]] = piece.get("kind")
    bank = load_bank(args.bank) if args.bank else None
    params, leaf_ids, from_bank = {}, {}, []

    producers = {}
    if args.handover == "written":
        # Every record whose target writes a note, by note name: the producer of that note's block.
        with open(args.records) as stream:
            for line in stream:
                if "$write" in line:  # escaped inside the arguments string in the raw line
                    record = json.loads(line)
                    for name in target_writes(record):
                        if name in producers:raise ValueError('duplicate producer for '+name)
                        producers[name]={**record,'_active_write_name':name}

    backbone, heads = engine.backbone, engine.heads
    backbone.ffn_chunk_tokens = args.ffn_chunk_tokens
    backbone.checkpoint_layers = args.checkpoint_layers
    backbone.checkpoint_attention_only = args.checkpoint_attention_only
    active_staging = [None]
    selective_writer_replays = [0]
    from .memory_estimator import (AdaptiveGraphMemory, backbone_memory_layout,
                                   geometry_bytes, producer_geometry_bytes,
                                   writer_batch_kind)
    shared_kv_prefix = bool(args.checkpoint_layers and backbone.fast and
                            getattr(backbone, 'attention_checkpoint_prefixes', False))
    geometry_version = 'shared-prefix-v1' if shared_kv_prefix else 'full-prefix-v1'
    native_gold_tape = args.writer_supervision == 'native-value' and bool(
        args.writer_text_weight or (args.stop_supervision == 'gold-native-boundary' and args.stop_weight))
    if native_gold_tape:
        geometry_version += ':native-gold-tape-v1'
    plain_layers = (sum(not backbone.is_attention(i) for i in range(backbone.num_layers))
                    if args.checkpoint_attention_only else 0)
    if plain_layers:
        geometry_version += ':attention-only-v1'
    # A new gradient replay changes retained tape and transient branch geometry.
    # Never inherit earlier joint-fit routes across that resource boundary.
    geometry_version += (f':writer-{heads.profile}:{args.sketch_gradient}:k{heads.cutoff}'
                         f':local-group{args.local_stage_batch_size if args.sketch_gradient == "local_stage" else 1}')
    if args.sketch_gradient == 'local_stage':
        geometry_version += ':local-stage-geometry-v1'
    memory_estimator = AdaptiveGraphMemory(resumed.get('memory_estimator') if resumed else None,
                                           geometry_version=geometry_version)
    memory_layout = backbone_memory_layout(
        backbone, checkpointed=args.checkpoint_layers,
        shared_kv_prefix=shared_kv_prefix, uncheckpointed_layers=plain_layers,
        stage_group_size=args.local_stage_batch_size if args.sketch_gradient == "local_stage" else 0,
        sketch_cutoff=heads.cutoff)
    geometry_cache = {}
    from .recurrence import ProducerMemo, is_acyclic, independent_frontier, dependency_frontiers
    dependencies = {name: (reads(record) | set(handover_notes(record))) - target_writes(record)
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
        return source_vector_length(len(engine._tokens(text)), args.tokens_per_vector, heads.max_length)

    native_sources = {}
    def producer_source(name, producer):
        source = handover_notes(producer).get(name)
        if args.writer_length_policy == 'source-text' or source is None:
            return source
        if name not in native_sources:
            from ..serve.chat import write_value_text
            tool, before, argument, own = write_site(producer)
            if own != name:
                raise ValueError('producer source does not match its native write site')
            value_type = write_value_type(producer)
            value = json.loads(source) if value_type == 'unknown' else source
            native_sources[name] = write_value_text(
                lambda m, g: engine.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=g),
                tool, before, argument, value, value_type)
        return native_sources[name]

    if args.tokens_per_vector:
        # Fail configuration before any update, rather than discover a clipped
        # producer only when a later reader reaches it.
        for name, producer in producers.items():
            source_length(producer_source(name, producer))

    def write(messages, tools, prefix, leaves, source: str | None = None, resource_choice=None, gold_stop_supervised=False):
        previous = backbone.checkpoint_attention_only
        try:
            return write_impl(messages, tools, prefix, leaves, source, resource_choice, gold_stop_supervised)
        finally:
            backbone.checkpoint_attention_only = previous

    def write_impl(messages, tools, prefix, leaves, source: str | None = None, resource_choice=None, gold_stop_supervised=False):
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
        if (args.staged_checkpoint_attention_only and active_staging[0] is not None
                and torch.is_grad_enabled() and args.device.startswith('cuda')
                and getattr(backbone, 'attention_checkpoint_prefixes', False)):
            from .memory_estimator import selective_writer_fits
            choose = lambda: selective_writer_fits(
                context.shape[1], source_length(source) or heads.max_length, memory_layout,
                baseline=cuda_allocated_bytes(), budget=graph_budget,
                plain_layers=sum(not backbone.is_attention(i) for i in range(backbone.num_layers)))
            backbone.checkpoint_attention_only = resource_choice.resolve(choose) if resource_choice else choose()
            selective_writer_replays[0] += int(backbone.checkpoint_attention_only)
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
            written = write_generated(backbone, heads, pre, args.sketch_gradient, sketch_target_backbone_scale=args.sketch_target_backbone_scale, local_stage_batch_size=args.local_stage_batch_size, length=target)
            if args.stop_weight and not gold_stop_supervised and torch.is_grad_enabled():
                boundary_terms.append(args.stop_weight * stop_boundary_loss(written))
        else:
            written = write_generated(backbone, heads, pre, args.sketch_gradient, sketch_target_backbone_scale=args.sketch_target_backbone_scale, local_stage_batch_size=args.local_stage_batch_size, sample=bool(args.stop_pg), generator=stop_generator)
        if args.sketch_target_weight and torch.is_grad_enabled():
            if written.sketch_target_loss is None:
                raise ValueError('sketch self-target requires one_step or local_stage writes')
            boundary_terms.append(args.sketch_target_weight * written.sketch_target_loss)
        # Gold lengths are already known on the host. Reading them back from
        # CUDA adds a completion wait without adding any information.
        n = target if target is not None else int(written.lengths[0])
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
    writer_batches: list[int] = []
    stop_terms: list = []  # (log-probability of the stop decisions, length) of this record's writes
    boundary_terms: list = []  # stop-boundary losses of this record's source-sized writes
    stop_generator = torch.Generator().manual_seed(args.seed)
    write_choice = random.Random(args.seed)
    baseline = {"value": None}

    prefixes = {}

    def site_prefix(record):
        """The forced reply before the written argument, in the model's own rendering of the producer's call."""
        call, before, argument, _ = write_site(record)
        value_type = write_value_type(record) if not engine.heads.read_markers else "string"
        own=write_site(record)[3]
        code_prefix=inline_write_prefix(record,own)
        key = (call, json.dumps(before, sort_keys=True), argument, value_type, code_prefix)
        if key not in prefixes:
            prefixes[key]=native_writer_prefix(record,
                lambda m,g:engine.tokenizer.apply_chat_template(m,tokenize=False,add_generation_prompt=g),value_type=value_type)
        return prefixes[key]

    def prepare_note(name, leaves, depth, visiting, memo):
        producer = producers[name]
        names, payloads = {}, {}
        if depth < args.write_depth:
            # Depth counts writes, once per edge. Previously incrementing here
            # AND in written_values silently made depth3 only two write layers.
            names, payloads = written_values(producer, leaves, depth, visiting + (name,), memo)
        messages = render(producer["messages"], lambda n: {"type": "neuralese", "id": leaf_ids[n]}, handover_notes(producer),
                          names, names)
        from .memory_estimator import ReplayResourceChoice
        resource_choice = ReplayResourceChoice()
        def replay():
            begin = len(boundary_terms)
            result = write(messages, producer.get("tools"), site_prefix(producer),
                           resolve_values({**leaves, **payloads}), source=producer_source(name, producer),
                           resource_choice=resource_choice, gold_stop_supervised=args.stop_supervision == 'gold-native-boundary')
            terms = boundary_terms[begin:]
            del boundary_terms[begin:]
            return result, terms
        def gold_replay():
            if producer.get('split') != 'train' or producer.get('training_admission', {}).get('approved') is not True:
                raise ValueError('producer gold supervision requires admitted training split')
            if args.writer_supervision == 'native-value':
                return session.supervised_continuation_loss(
                    messages, producer.get('tools'), site_prefix(producer), producer_source(name, producer),
                    resolve_values({**leaves, **payloads}), text_weight=args.writer_text_weight,
                    stop_weight=args.stop_weight if args.stop_supervision == 'gold-native-boundary' else 0.)
            target = producer_text_target(producer, texts, names)
            return args.writer_text_weight * session.supervised_text_loss(
                {'messages': messages, 'tools': producer.get('tools'), 'target': target},
                resolve_values({**leaves, **payloads}))
        auxiliary = gold_replay if (args.writer_text_weight or (args.stop_supervision == 'gold-native-boundary' and args.stop_weight)) and torch.is_grad_enabled() else None
        return producer, messages, payloads, replay, auxiliary

    def note_payload(name, leaves, depth=1, visiting=(), memo=None):
        """One producer and its recursively prepared dependencies, sharing the same writer surface."""
        if memo is None:
            memo = ProducerMemo(share_producers)
        def compute():
            _, _, _, replay, auxiliary = prepare_note(name, leaves, depth, visiting, memo)
            if active_staging[0] is not None:
                return active_staging[0].add(replay, auxiliary=auxiliary)
            value, terms = replay()
            if auxiliary is not None:
                terms = [sum(terms) + auxiliary()]
            boundary_terms.extend(terms)
            return value
        return memo.write(name, depth, compute)

    def producer_frontier(chosen, leaves, depth, visiting, memo):
        """Prepare dependencies first, then admit pinned independent tensor groups."""
        if not independent_frontier(chosen, dependencies):
            raise ValueError('dependent producers cannot share a tensor batch')
        jobs = []
        for name in chosen:
            if (name, depth) in memo.values:
                continue
            producer, messages, payloads, replay, auxiliary = prepare_note(name, leaves, depth, visiting, memo)
            prompt = render_messages(messages, producer.get('tools'), engine._template, engine.specials)
            items = session._items(prompt.segments, prompt.blocks, prompt.escape_nonce)
            items += [('tok', t) for t in engine._tokens(site_prefix(producer))]
            scope = {**leaves, **payloads}
            resolved = resolve_values(scope)
            width = sum(1 if kind == 'tok' else resolved[value].shape[0] if value in resolved
                        else engine.lookup(value).length for kind, value in items)
            jobs.append({'name': name, 'items': items, 'scope': scope, 'width': width,
                         'length': source_length(producer_source(name, producer)),
                         'replay': replay, 'auxiliary': auxiliary})
        while jobs:
            count = min(args.producer_batch_size, len(jobs))
            # Joint execution retains preceding tapes. Staged execution releases
            # them and can use its separately declared frontier envelope.
            frontier_budget = batch_graph_budget if active_staging[0] is not None else graph_budget
            baseline_bytes = cuda_allocated_bytes() if args.device.startswith('cuda') else 0
            while count > 1:
                group = tuple(jobs[:count])
                width, vectors = max(j['width'] for j in group), max(j['length'] for j in group)
                raw = count * geometry_bytes(width, vectors, **{**memory_layout, 'uncheckpointed_layers': 0})
                predicted = memory_estimator.predict(writer_batch_kind(count), width, vectors, raw)
                if not frontier_budget or baseline_bytes + predicted <= frontier_budget:
                    break
                count -= 1
            group = tuple(jobs[:count])
            del jobs[:count]
            if count == 1:
                job = group[0]
                if active_staging[0] is not None:
                    node = active_staging[0].add(job['replay'], auxiliary=job['auxiliary'])
                else:
                    node, terms = job['replay']()
                    if job['auxiliary'] is not None:
                        terms = [sum(terms) + job['auxiliary']()]
                    boundary_terms.extend(terms)
                memo.write(job['name'], depth, lambda node=node: node)
                continue
            width, vectors = max(j['width'] for j in group), max(j['length'] for j in group)
            plain = sum(not backbone.is_attention(i) for i in range(backbone.num_layers))
            selective_raw = count * geometry_bytes(width, vectors, **{**memory_layout, 'uncheckpointed_layers': plain})
            selective_kind = writer_batch_kind(count, selective=True)
            selective_estimate = memory_estimator.predict(selective_kind, width, vectors, selective_raw)
            selective = bool(active_staging[0] is not None and args.staged_checkpoint_attention_only and args.checkpoint_layers and
                             (count <= 2 or len(memory_estimator.calibration(selective_kind, width, vectors)) >= 3) and
                             (not frontier_budget or baseline_bytes + selective_estimate <= frontier_budget))
            raw = selective_raw if selective else count * geometry_bytes(width, vectors, **{**memory_layout, 'uncheckpointed_layers': 0})
            kind = writer_batch_kind(count, selective=selective)
            def compute_batch(group=group, selective=selective, frontier_budget=frontier_budget):
                # Membership and checkpoint policy are captured once.
                # Rebuild embedded scope on replay, retaining all child VJPs.
                previous = backbone.checkpoint_attention_only
                backbone.checkpoint_attention_only = selective
                selective_writer_replays[0] += len(group) if selective else 0
                try:
                    contexts = [session._embed_items(j['items'], resolve_values(j['scope']))[0] for j in group]
                    if any(c.shape[0] != j['width'] for c, j in zip(contexts, group)):
                        raise RuntimeError('producer batch scope geometry changed during replay')
                    if args.detach_write_context:
                        contexts = [c.detach() for c in contexts]
                    from .execution import prefill_write_contexts
                    with graph_memory_budget(frontier_budget if args.device.startswith('cuda') else 0):
                        pre = prefill_write_contexts(backbone, heads, contexts)
                        # Validate/max the known lengths on CPU; unroll_write
                        # transfers them once for its returned row metadata.
                        sizes = torch.tensor([j['length'] for j in group])
                        written = write_generated(backbone, heads, pre, args.sketch_gradient, sketch_target_backbone_scale=args.sketch_target_backbone_scale, local_stage_batch_size=args.local_stage_batch_size, lengths=sizes)
                    write_context_lengths.extend(j['width'] for j in group)
                    lengths.extend(j['length'] for j in group)
                    writer_batches.append(len(group))
                    penalties = [[] for _ in group]
                    if args.sketch_target_weight and torch.is_grad_enabled():
                        row_losses = written.sketch_target_loss_by_row
                        if row_losses is None or row_losses.shape != (len(group),):
                            raise RuntimeError('batched sketch self-target did not return one loss per producer row')
                        penalties = [[args.sketch_target_weight * row_losses[row]] for row in range(len(group))]
                    return [written.payload[row, :j['length']] for row, j in enumerate(group)], penalties
                finally:
                    backbone.checkpoint_attention_only = previous
            def observe_batch(values, retained, kind=kind, width=width, vectors=vectors, raw=raw):
                memory_estimator.observe(kind, width, vectors, raw, retained)
            if active_staging[0] is not None:
                nodes = active_staging[0].add_batch(compute_batch, auxiliaries=[j['auxiliary'] for j in group],
                                                  observe=observe_batch)
            else:
                before = retained_tape_bytes() if args.device.startswith('cuda') else 0
                nodes, penalty_rows = compute_batch()
                if args.device.startswith('cuda'):
                    observe_batch(nodes, max(0, retained_tape_bytes() - before))
                for job, terms in zip(group, penalty_rows):
                    if job['auxiliary'] is not None:
                        terms = [sum(terms) + job['auxiliary']()]
                    boundary_terms.extend(terms)
            for job, node in zip(group, nodes):
                memo.write(job['name'], depth, lambda node=node: node)
        return {name: memo.values[(name, depth)] for name in chosen}

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

    def written_values(record, leaves, depth=0, visiting=(), memo=None, reader_only=False):
        """Blocks written for this record this step: handoffs it reads or shows (notes, child results), digests in its
        listing. Inside a producer (depth > 0) its own target's value is not one of them. Returns (name → placeholder
        ID, placeholder ID → payload)."""
        names, payloads = {}, {}
        if memo is None:
            memo = ProducerMemo(share_producers)
        if args.handover == "written":
            own = target_writes(record) if depth or reader_only else set()
            visible = reads(record) if reader_only else reads(record) | set(handover_notes(record))
            chosen = [name for name in sorted(visible)
                      if name in producers and name not in own and name not in visiting]
            if args.max_writes and len(chosen) > args.max_writes:
                chosen = sorted(write_choice.sample(chosen, args.max_writes))
            frontier = None
            if (args.producer_batch_size > 1 and share_producers and len(chosen) > 1
                    and (active_staging[0] is not None or (args.joint_producer_batching and torch.is_grad_enabled()))):
                frontier = {}
                for ready in dependency_frontiers(chosen, dependencies):
                    frontier.update(producer_frontier(ready, leaves, depth + 1, visiting, memo))
            for name in chosen:
                names[name] = placeholder(name)
                payloads[names[name]] = frontier[name] if frontier is not None else note_payload(name, leaves, depth + 1, visiting, memo)
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

    train, held_pool, skipped = [], [], {"long": 0, "no-target": 0}
    with open(args.records) as stream:
        for line in stream:
            record = json.loads(line)
            if not record.get("target"):
                skipped["no-target"] += 1
                continue
            if args.only_handover and not (reads(record) or handover_notes(record)):
                continue
            bucket = held_pool if record.get("split") == "test" else train
            if bucket is train and len(train) >= args.train:
                continue
            if prompt_tokens(record) > args.max_tokens:
                skipped["long"] += 1
                continue
            bucket.append(record)
    from .trajectory_probe import select_held, select_paired_held, source_groups
    probe_policy = 'typed-factual-reciprocal-readers-v1'
    held = select_held(held_pool, args.eval)
    paired_held, held_probe_accounting = select_paired_held(held_pool, args.eval, producers, piece_kinds)
    paired_train, train_probe_accounting = select_paired_held(train, args.eval, producers, piece_kinds)
    probe_selection = {'policy': probe_policy, 'ce_selected_ids': [r['id'] for r in held],
                       'held_pool_rows': len(held_pool), 'held_written': held_probe_accounting,
                       'train_written': train_probe_accounting,
                       'scope': 'Declared typed factual-disjoint reciprocal readers only; exclusions explicit. '
                                'Forced native lengths; not autonomous stop or broad task quality qualification.'}
    probe_selection_hash = hashlib.sha256(json.dumps(probe_selection, sort_keys=True).encode()).hexdigest()
    (out / 'eval-selection.json').write_text(json.dumps(probe_selection, indent=2) + '\n')
    # Soft parameters for the names the selected records use (a corpus has thousands of instructions texts).
    used_names = {part["name"] for record in train + held + paired_held + paired_train + list(producers.values()) for message in record["messages"]
                  if isinstance(message.get("content"), list) for part in message["content"] if part["type"] == "soft"}
    if args.digest == "written":
        used_names.add("prompt:digest")
    from .trajectory_state import soft_initialization, resumed_initial_rows, iteration_rng_state, restore_iteration_rng
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

    from .backbone_policy import configure_backbone_training, backbone_trainable_state, resolve_backbone_policy
    args.backbone_training=resolve_backbone_policy(engine.backbone,args.backbone_training)
    if args.backbone_training=='lora' and args.rank<1:
        raise ValueError('explicit LoRA policy requires --rank positive')
    backbone_named = configure_backbone_training(engine.backbone,args.backbone_training,rank=args.rank or 16)
    lora = [parameter for _,parameter in backbone_named]
    lora_names = ([name for name,_ in backbone_named]
                  if args.backbone_training in ('full','qat') else None)
    qat_named = backbone_named if args.backbone_training=='qat' else []
    backbone_lr = args.backbone_lr if args.backbone_training=='full' else args.lora_lr

    def policy_state():
        return backbone_trainable_state(backbone_named)
    if args.checkpoint_elide_rng:
        if not hasattr(backbone, 'elide_checkpoint_rng'):
            raise ValueError('checkpoint RNG elision is qualified only for the native LFM2 port')
        backbone.elide_checkpoint_rng()
    if args.crisp_weight and not lora:
        raise ValueError('crisp SFT requires a trainable backbone policy')
    # The writer's own modules (feedback, content projection) learn from the readers of what they write.
    # Native final normalization is a frozen reference. At full depth the
    # causal feedback is already exact; learn payload/stop without corrupting it.
    head_params = [p for name, p in heads.named_parameters()
                   if not (not heads.read_markers and (name.startswith('feedback.final_norm.') or name.startswith('content.reference.') or
                           (heads.cutoff == backbone.num_layers and name.startswith('feedback.'))))] if args.heads_lr and (args.handover == "written" or args.digest == "written") else []
    if args.train_control_rows:
        head_params += [backbone.control_rows]
        if not getattr(backbone, 'tied', True):
            head_params += [backbone.control_head_rows]
    for p in head_params:
        p.requires_grad_(True)
    optimizer = trajectory_optimizer(args.optimizer, params, lora, head_params,
                                     vocab_size=backbone.embedding_weight.shape[0], lr=args.lr,
                                     lora_lr=backbone_lr, heads_lr=args.heads_lr,
                                     embedding_ids={id(backbone.control_rows), id(getattr(backbone, 'control_head_rows', backbone.control_rows))} | {id(p) for m in heads.modules() if isinstance(m, torch.nn.Embedding) for p in m.parameters()},
                                     lora_names=lora_names)
    if resumed is not None:
        if set(params) != set(resumed['params']):
            raise ValueError('recurrence soft-parameter names changed')
        with torch.no_grad():
            for name, value in resumed['params'].items():
                params[name].copy_(value.to(params[name]))
            for name, value in {**resumed.get('lora', {}), **resumed.get('backbone_trainables', {})}.items():
                parameters=dict(backbone.hf.named_parameters())
                if name not in parameters or parameters[name].shape != value.shape:
                    raise ValueError('resumed backbone parameter mismatch: '+name)
                parameters[name].copy_(value.to(parameters[name]))
        with torch.no_grad():
            backbone.control_rows.copy_(resumed['control_rows'].to(backbone.control_rows))
            if not getattr(backbone, 'tied', True):
                backbone.control_head_rows.copy_(resumed['control_head_rows'].to(backbone.control_head_rows))
        heads.load_state_dict(resumed['heads'])
        optimizer.load_state_dict(resumed['optimizer'])
        init = {k: v.to(params[k]) for k, v in resumed['init'].items()}
    if new_continuation and args.content_residual_initialization == 'fresh-zero':
        from .trajectory_state import initialize_content_residual
        reset = initialize_content_residual(heads, optimizer)
        (out / 'curriculum-initialization.json').write_text(json.dumps({
            'schema': 'natlang.content-residual-initialization/1', 'reset_parameters': reset,
            'optimizer_reset_scope': reset, 'other_optimizer_and_rng_preserved': True,
            'parent_sha256': identity['continuation']['checkpoint_sha256'],
            'initial_content_projection_zero': all(bool(torch.count_nonzero(p) == 0) for p in heads.content.proj.parameters()),
            'semantic_qualification_inherited': False}, indent=2) + '\n')
    if not heads.read_markers:
        heads.configure_frozen_reference()

    def count_embedding(items, dimensions):
        return sum(1 if kind == 'tok' else dimensions[value] + 2 for kind, value in items)

    def geometry_plan(record):
        dimensions = {block: params[name].shape[0] for name, block in leaf_ids.items()}
        features, memo = [], {}
        def visit(current, depth=0, visiting=()):
            own = target_writes(current) if depth else set()
            chosen = [name for name in sorted(reads(current) | set(handover_notes(current)))
                      if name in producers and name not in own and name not in visiting]
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
                    vectors = source_length(producer_source(name, producer)) or heads.max_length
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
                           producer_geometry_bytes(context, vectors, memory_layout, native_gold=native_gold_tape,
                                                   vocab_size=backbone.embedding_weight.shape[0])) for context, vectors in features)
        reader_raw = geometry_bytes(reader_context, 0, **memory_layout, target_tokens=target_tokens,
                                    vocab_size=backbone.embedding_weight.shape[0])
        reader_bytes = memory_estimator.predict('reader', reader_context, target_tokens, reader_raw)
        from .memory_estimator import local_stage_kv_workspace
        max_kv_width = max([2 * backbone.layers[i].self_attn.k_proj.out_features
                            for i in range(backbone.num_layers) if backbone.is_attention(i)] or [0])
        workspace = max([local_stage_kv_workspace(c, n, memory_layout['stage_group_size'],
                         max_layer_kv_width=max_kv_width, dtype_bytes=memory_layout['dtype_bytes'])
                         for c, n in features] or [0])
        return {'writers': features, 'reader_context': reader_context, 'target_tokens': target_tokens,
                'reader_raw': reader_raw, 'tape_bytes': writer_bytes + reader_bytes,
                'branch_kv_workspace_bytes': workspace}

    reader_geometry = [None]
    offload_stats = {'offloaded_bytes': 0, 'live_offloaded_bytes': 0, 'peak_offloaded_bytes': 0}
    def retained_tape_bytes():
        # The predictor admits joint graphs without offload. Include CPU-stored
        # tape in observations instead of teaching it a falsely smaller cost.
        return cuda_allocated_bytes() + offload_stats.get('live_offloaded_bytes', 0)
    def observe_writer(value, retained_bytes):
        if args.staged_checkpoint_attention_only:
            # Staged replays can use different tapes. Never contaminate the
            # all-checkpointed joint admission profile with these measurements.
            return
        context, vectors = write_context_lengths[-1], value.shape[0]
        raw = producer_geometry_bytes(context, vectors, memory_layout, native_gold=native_gold_tape,
                                      vocab_size=backbone.embedding_weight.shape[0])
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
        reader_before = retained_tape_bytes() if args.device.startswith('cuda') else 0
        loss = session.supervised_text_loss(
            {"messages": messages, "tools": record.get("tools"), "target": target}, leaves,
            teacher_messages=crisp_messages(record["messages"], texts, handover_notes(record)) if distill else None,
            distill_weight=distill, context_weight=args.context_weight if training_objective else 0.,
            feedback_weight=args.feedback_weight)
        if reader_geometry[0] and torch.is_grad_enabled() and args.device.startswith('cuda'):
            plan = reader_geometry[0]
            memory_estimator.observe('reader', plan['reader_context'], plan['target_tokens'], plan['reader_raw'],
                                     max(0, retained_tape_bytes() - reader_before))
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
            for record in _progress(held, label):
                try:
                    values.append(float(loss_of(record, leaves, soft, training_objective=False)))
                except RequestError:
                    pass
        return {"label": label, "cross_entropy": sum(values) / max(1, len(values)), "n": len(values)}

    def evaluate_written(label, leaves, records, accounting):
        """Pure reader CE with own and proven factual-disjoint typed donor blocks.

        Donors are fixed before execution. A failed writer never changes donor
        selection, drops an eligible row silently, or cycles one block into slots.
        """
        stop_terms.clear()
        boundary_terms.clear()
        sites, write_errors, reader_errors, missing_donors = {}, [], [], []
        probe_memo = ProducerMemo(share_producers)
        with torch.no_grad():
            for record in _progress(records, label):
                try:
                    names, payloads = written_values(record, leaves, memo=probe_memo, reader_only=True)
                    if not payloads:
                        raise RequestError('paired_reader_missing_payload', 'selected paired reader produced no read payloads')
                    sites[record['id']] = (record, names, payloads)
                except RequestError as error:
                    write_errors.append({'id': record['id'], 'error': str(error)})
        result = {'label': label, 'policy': probe_policy, 'n': 0, 'expected_n': len(records),
                  'selection_sha256': probe_selection_hash,
                  'coverage': accounting, 'case_scores': [],
                  'producer_cache_hits': probe_memo.hits, 'producer_cache_entries': len(probe_memo.values),
                  'write_errors': write_errors, 'reader_errors': reader_errors, 'missing_donors': missing_donors,
                  'qualification_scope': probe_selection['scope']}
        own, shuffled, groups = [], [], {}
        with torch.no_grad():
            for record, names, payloads in sites.values():
                rid = record['id']
                plan = accounting['selected_mappings'][rid]
                donor_id = plan['donor_id']
                if donor_id not in sites:
                    missing_donors.append({'id': rid, 'donor_id': donor_id, 'reason': 'fixed donor write failed'})
                    continue
                donor_record, donor_names, donor_payloads = sites[donor_id]
                mapping = plan['donor_payload_to_recipient_payload']
                if set(mapping) != set(donor_names) or set(mapping.values()) != set(names):
                    raise RuntimeError('paired reader slot proof disagrees with actual resolved reads')
                swapped = {names[recipient]: donor_payloads[donor_names[donor]]
                           for donor, recipient in mapping.items()}
                if set(swapped) != set(payloads):
                    raise RuntimeError('paired reader payload bijection changed during execution')
                def reader(values):
                    return float(session._term({'kind': 'crossEntropy', 'messages': soft_messages(record, names),
                                                'tools': record.get('tools'), 'target': target_of(record, names)},
                                               {**leaves, **values}))
                try:
                    a, b = reader(payloads), reader(swapped)
                except RequestError as error:
                    reader_errors.append({'id': rid, 'error': str(error)})
                    continue
                own.append(a)
                shuffled.append(b)
                factual = sorted(source_groups(record))
                key = tuple(factual)
                groups.setdefault(key, []).append((a, b))
                result['case_scores'].append({'id': rid, 'source_groups': factual, 'donor_id': donor_id,
                    'donor_source_groups': sorted(source_groups(donor_record)), 'payload_mapping': mapping,
                    'written': a, 'shuffled': b, 'margin': b - a,
                    'slot_lengths': [{'recipient': recipient, 'donor': donor,
                                     'own': int(payloads[names[recipient]].shape[0]),
                                     'donor_length': int(donor_payloads[donor_names[donor]].shape[0])}
                                    for donor, recipient in sorted(mapping.items())]})
        stop_terms.clear()
        boundary_terms.clear()
        if own:
            group_means = [{'source_groups': list(key), 'rows': len(values),
                            'written': sum(a for a, _ in values) / len(values),
                            'shuffled': sum(b for _, b in values) / len(values)}
                           for key, values in sorted(groups.items())]
            result.update({'written': sum(own) / len(own), 'shuffled': sum(shuffled) / len(shuffled),
                           'n': len(own), 'written_better': sum(a < b for a, b in zip(own, shuffled)) / len(own),
                           'factual_group_set_means': group_means,
                           'factual_group_set_written_mean': sum(g['written'] for g in group_means)/len(group_means),
                           'factual_group_set_shuffled_mean': sum(g['shuffled'] for g in group_means)/len(group_means)})
        return result

    leaves = {leaf_ids[name]: p for name, p in params.items()}
    report = dict(resumed['initial_report']) if resumed is not None else {"crisp": evaluate("crisp", {}, soft=False), "soft-init": evaluate("soft-init", leaves)}
    if resumed is None and (args.handover == "written" or args.digest == "written"):
        report["written-init"] = evaluate_written("written-init", leaves, paired_held, held_probe_accounting)
        report["written-init-train"] = evaluate_written("written-init-train", leaves, paired_train, train_probe_accounting)
    if resumed is not None and resumed.get('probe_selection_sha256') != probe_selection_hash:
        # Evaluate the new scope before restoring RNG state below. Historical
        # initial scores remain explicitly historical, not re-labelled as new.
        baseline_eval = {'completed_updates': resumed['step'], 'event': 'probe_scope_baseline',
                         'selection_sha256': probe_selection_hash,
                         'soft': evaluate('resume-scope-soft', leaves)}
        if args.handover == 'written' or args.digest == 'written':
            baseline_eval['written'] = evaluate_written('resume-scope-written', leaves, paired_held, held_probe_accounting)
        report['resume_scope_baseline'] = baseline_eval
        report['historical_initial_scope'] = 'Earlier initial reports retain their original selection scope.'
        with (out / 'eval.jsonl').open('a') as baseline_log:
            baseline_log.write(json.dumps(baseline_eval) + '\n')
    print(json.dumps(report), flush=True)
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
    graph_routes = dict(resumed.get('graph_routes', {})) if resumed and memory_estimator.joint_routes_compatible else {}
    trainables = list(params.values()) + lora + head_params
    stop_requested = [False]
    previous_handlers = {sig: signal.signal(sig, lambda *_: stop_requested.__setitem__(0, True))
                         for sig in (signal.SIGTERM, signal.SIGINT)}
    from .trajectory_state import compatible_best_evaluation
    selection_signature = {'files': identity['files'], 'port_profile': heads.profile,
                           'content_transport': heads.content.transport, 'sketch_gradient': args.sketch_gradient,
                           'writer_length_policy': args.writer_length_policy, 'writer_supervision': args.writer_supervision, 'stop_supervision': args.stop_supervision,
                           'tokens_per_vector': args.tokens_per_vector, 'max_write_vectors': heads.max_length,
                           'write_depth': args.write_depth, 'max_writes': args.max_writes,
                           'write_curriculum': args.write_curriculum, 'probe_policy': probe_policy,
                           'probe_selection_sha256': probe_selection_hash}
    inherited_best = resumed.get('best_evaluation') if resumed else None
    best_evaluation = compatible_best_evaluation(inherited_best, selection_signature)
    best_history = list(resumed.get('best_evaluation_history', [])) if resumed else []
    if inherited_best is not None and best_evaluation is None:
        if inherited_best not in best_history:
            best_history.append(inherited_best)
        print(json.dumps({'status': 'candidate_selection_regime_changed',
                          'inherited_candidate_preserved_in_history': True,
                          'selection_signature': selection_signature}), flush=True)
    host_gc_seconds = 0.0
    host_gc_calls = 0

    def collect_graph_cycles():
        # Host wall time only: do not synchronize CUDA or change collection cadence.
        nonlocal host_gc_seconds, host_gc_calls
        started_gc = time.perf_counter()
        try:
            return gc.collect()
        finally:
            host_gc_seconds += time.perf_counter() - started_gc
            host_gc_calls += 1

    def save_training_state(step, destination=None):
        started_save = time.perf_counter()
        destination = destination or checkpoint_path
        atomic_checkpoint(destination, {
            'schema': 'natlang.neuralese_recurrence_checkpoint/1', 'identity': identity, 'graph_routes': graph_routes, 'memory_estimator': memory_estimator.state_dict(),
            'execution_policy': {'checkpoint_layers': args.checkpoint_layers,
                                 'checkpoint_preserve_rng': getattr(backbone, 'checkpoint_preserve_rng', True),
                                 'producer_batch_size': args.producer_batch_size,
                                 'joint_producer_batching': args.joint_producer_batching,
                                 'token_cache_mib': args.token_cache_mib,
                                 'producer_batch_memory_gb': args.producer_batch_memory_gb,
                                 'checkpoint_attention_only': args.checkpoint_attention_only,
                                 'staged_checkpoint_attention_only': args.staged_checkpoint_attention_only,
                                 'activation_offload_gb': args.activation_offload_gb,
                                 'geometry_version': geometry_version},
            'step': step, 'cursor': cursor, 'errors': errors, 'used': sorted(used), 'best_evaluation': best_evaluation, 'best_evaluation_history': best_history,
            'params': {k: v.detach().cpu() for k, v in params.items()}, 'texts': texts,
            'control_rows': backbone.control_rows.detach().cpu(),
            **({'control_head_rows': backbone.control_head_rows.detach().cpu()} if not getattr(backbone, 'tied', True) else {}),
            'port_config': {'cutoff': heads.cutoff, 'max_length': heads.max_length, **heads.port_config()},
            'heads': heads.state_dict(), 'lora': lora_state(backbone), 'optimizer': optimizer.state_dict(),
            'backbone_training': args.backbone_training,
            **({'maple_qat': True} if qat_named else {}),
            **({'backbone_trainables': policy_state()} if args.backbone_training in ('full','qat') else {}),
            'init': {k: v.detach().cpu() for k, v in init.items()}, 'initial_report': report,
            'probe_selection_sha256': probe_selection_hash,
            'baseline': baseline, 'python_rng': random.getstate(), 'write_rng': write_choice.getstate(),
            'stop_rng': stop_generator.get_state(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else []})
        # Includes state construction, device copies, serialization and durable I/O.
        # This is a blocking phase measurement, not CPU-exclusive or CUDA kernel time.
        phase = {'event': 'checkpoint_saved', 'completed_updates': step,
                 'path': str(destination), 'blocking_seconds': time.perf_counter() - started_save}
        with (out / 'host-phases.jsonl').open('a') as phase_log:
            phase_log.write(json.dumps(phase) + '\n')
    save_training_state(start_step)
    # The model, fixed corpus, optimizer and imported libraries live until this
    # dedicated trainer process exits. Collect completed setup/probe cycles first,
    # then exclude this static population from every staged graph collection.
    # Dynamic training graphs created below remain tracked/collectible. Do not
    # unfreeze an existing library baseline or freeze after an update's graph.
    setup_collected = gc.collect()
    frozen_before = gc.get_freeze_count()
    gc.freeze()
    print(json.dumps({'event': 'static_gc_baseline_frozen', 'setup_collected': setup_collected,
                      'previous_frozen_objects': frozen_before, 'frozen_objects': gc.get_freeze_count(),
                      'scope': 'process-lifetime setup only; future graph collection unchanged'}), flush=True)
    with torch.enable_grad():
        for step in range(start_step, args.steps):
            step_started = time.time()
            step_gc_seconds, step_gc_calls = host_gc_seconds, host_gc_calls
            step_lengths_start = len(lengths)
            step_batches_start = len(writer_batches)
            step_contexts_start = len(write_context_lengths)
            selective_writer_replays[0] = 0
            step_cursor, step_errors, step_used = cursor, errors, set(used)
            step_rng = iteration_rng_state(write_choice, stop_generator, baseline, cuda=args.device.startswith('cuda'))
            optimizer.zero_grad(set_to_none=True)
            losses = []
            crisp_losses = []
            released_graph_bytes = 0
            offload_stats = {'offloaded_bytes': 0, 'live_offloaded_bytes': 0, 'peak_offloaded_bytes': 0}
            staged_nodes, replay_error = 0, 0.0
            discarded_writer_batches, discarded_writer_rows = 0, 0
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
                    baseline_bytes = cuda_allocated_bytes() if args.device.startswith('cuda') else 0
                    raw_prediction = baseline_bytes + plan['tape_bytes'] + plan['branch_kv_workspace_bytes']
                    predicted = memory_estimator.adjust_joint(plan, raw_prediction)
                    mode = 'staged' if predicted > graph_budget or graph_routes.get(record['id'], 0) >= graph_budget else 'joint'
                    with (out / 'memory-routing.jsonl').open('a') as routing_log:
                        routing_log.write(json.dumps({'step': step, 'record_id': record['id'], 'mode': mode,
                            'estimated_gib': predicted / 2**30, 'graph_budget_gib': graph_budget / 2**30,
                            'write_sites': len(plan['writers']), 'writer_geometry': plan['writers'],
                            'sketch_gradient': args.sketch_gradient, 'local_stage_batch_size': args.local_stage_batch_size,
                            'branch_kv_workspace_gib': plan['branch_kv_workspace_bytes'] / 2**30}) + '\n')
                rng_before = (random.getstate(), write_choice.getstate(), stop_generator.get_state(),
                              torch.get_rng_state(), torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],
                              dict(baseline), len(lengths), len(write_context_lengths), len(writer_batches))
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
                            with (out / 'memory-routing.jsonl').open('a') as routing_log:
                                routing_log.write(json.dumps({'event': 'joint_failure', 'step': step,
                                    'record_id': record['id'], 'failure': failure,
                                    'observed_failed_peak_gib': joint_peak / 2**30,
                                    'estimated_gib': predicted / 2**30,
                                    'local_stage_batch_size': args.local_stage_batch_size}) + '\n')
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
                            discarded_writer_batches += len(writer_batches) - rng_before[8]
                            discarded_writer_rows += sum(writer_batches[rng_before[8]:])
                            del writer_batches[rng_before[8]:]
                            boundary_terms.clear()
                            stop_terms.clear()
                            collect_graph_cycles()
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
                        active_staging[0] = StagedWrites(observe=observe_writer, measure=retained_tape_bytes if args.device.startswith('cuda') else None, collect=collect_graph_cycles) if mode == 'staged' else None
                        from .memory import offload_attention_tensors
                        persistent = list(backbone.parameters()) + list(backbone.buffers()) + list(heads.parameters()) + list(heads.buffers()) + list(params.values())
                        with offload_attention_tensors(int(args.activation_offload_gb * 2**30), activations=True,
                                                      persistent_tensors=persistent) as offload_stats:
                            loss = loss_of(record, leaves) / args.batch
                            loss.backward()
                            losses.append(float(loss.detach()) * args.batch)
                            del loss
                            collect_graph_cycles()
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
                    collect_graph_cycles()
                    case_peak = torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0
                    step_peak_bytes = max(step_peak_bytes, case_peak)
                    if args.backward_policy == 'auto':
                        with (out / 'memory-routing.jsonl').open('a') as routing_log:
                            routing_log.write(json.dumps({'event': 'result', 'step': step, 'record_id': record['id'],
                                'mode': mode, 'observed_peak_gib': case_peak / 2**30,
                                'estimated_gib': predicted / 2**30}) + '\n')
                except RequestError as error:
                    if active_staging[0] is not None:
                        active_staging[0].clear()
                        active_staging[0] = None
                    errors += 1
                    # A skipped record contributes no gradient; say which and why (it was only counted).
                    print(json.dumps({'status': 'record_request_error', 'step': step, 'record_id': record['id'],
                                      'error': str(error)[:500]}), flush=True)
                    continue
                except (RuntimeError, AssertionError, ValueError) as error:
                    failure = {'status': 'training_out_of_memory' if isinstance(error, torch.OutOfMemoryError) else 'training_pre_update_failure',
                               'error_type': type(error).__name__, 'error': str(error),
                               'step': step, 'record_id': record['id'],
                               'curriculum': args.write_curriculum, 'batch': args.batch, 'backward_mode': mode,
                               'offload': dict(offload_stats)}
                    # No optimizer update occurs inside this accumulation loop.
                    # Keep all preceding updates and rewind this incomplete step,
                    # including every sampler, so resumption repeats its inputs.
                    optimizer.zero_grad(set_to_none=True)
                    if active_staging[0] is not None:
                        active_staging[0].clear()
                        active_staging[0] = None
                    cursor, errors, used = step_cursor, step_errors, step_used
                    restore_iteration_rng(step_rng, write_choice, stop_generator, baseline)
                    save_training_state(step)
                    failure['emergency_checkpoint'] = str(checkpoint_path)
                    failure['checkpoint_step'] = step
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
            entry = {"step": step, "iteration_index": step, "completed_updates": step + 1, "reader_record_ids": step_record_ids, "loss": sum(losses) / max(1, len(losses)), "seconds": round(time.time() - started),
                     "errors": errors, "backward_mode": mode, "staged_nodes": staged_nodes,
                     "host_gc_seconds": round(host_gc_seconds - step_gc_seconds, 6),
                     "host_gc_calls": host_gc_calls - step_gc_calls,
                     "replay_max_abs_error": replay_error, "selective_writer_replays": selective_writer_replays[0], "crisp_sft_loss": sum(crisp_losses) / max(1, len(crisp_losses)), "step_seconds": round(time.time() - step_started, 3), **({"writer_grad_norm": writer_grad} if head_params else {}),
                     **({"write_lengths": lengths[step_lengths_start:][-8:]} if len(lengths) > step_lengths_start else {}),
                     "writes_this_update": len(lengths) - step_lengths_start,
                     "writer_batch_calls_this_update": len(writer_batches) - step_batches_start,
                     "writer_batch_rows_this_update": sum(writer_batches[step_batches_start:]),
                     "max_writer_batch_rows_this_update": max(writer_batches[step_batches_start:], default=1),
                     "discarded_writer_batches_this_update": discarded_writer_batches,
                     "discarded_writer_rows_this_update": discarded_writer_rows,
                     "max_write_length_this_update": max(lengths[step_lengths_start:], default=0),
                     "write_capacity": heads.max_length}
            if args.device.startswith("cuda"):
                entry["peak_gb"] = round(max(step_peak_bytes, torch.cuda.max_memory_allocated()) / 2**30, 2)
                entry['released_graph_gib'] = round(released_graph_bytes / 2**30, 3)
                entry['largest_write_context_tokens'] = max(write_context_lengths[step_contexts_start:], default=0)
                entry['activation_offloaded_gib'] = round(offload_stats['offloaded_bytes'] / 2**30, 3)
                entry['activation_offload_peak_gib'] = round(offload_stats['peak_offloaded_bytes'] / 2**30, 3)
            if args.token_cache_mib:
                entry['token_cache'] = engine._token_cache.stats()
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
                        evaluation['written'] = evaluate_written('periodic-written', leaves, paired_held, held_probe_accounting)
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
                                       'selection_signature': selection_signature, 'checkpoint': str(out / 'best-checkpoint.pt'),
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
        report["written-trained"] = evaluate_written("written-trained", leaves, paired_held, held_probe_accounting)
        report["written-trained-train"] = evaluate_written("written-trained-train", leaves, paired_train, train_probe_accounting)
    if lengths:
        report["writes"] = {"count": len(lengths), "mean_length": sum(lengths) / len(lengths), "max_length": max(lengths)}
    if head_params or lora:
        from .adapters import adapter_layers
        source_metadata = torch.load(args.heads, map_location='cpu', weights_only=False, mmap=True) if args.heads else {}
        ranks = {int(p.shape[0]) for name, p in backbone.hf.named_parameters() if '.lora_A.' in name}
        if len(ranks) > 1:
            raise ValueError('cannot export mixed-rank adapters in one port checkpoint')
        atomic_checkpoint(out / 'heads.pt', {'heads': heads.state_dict(),
            'port_config': {**source_metadata.get('port_config', {}), 'cutoff': heads.cutoff,
                            'max_length': heads.max_length, **heads.port_config()},
            'control_rows': backbone.control_rows.detach().cpu(),
            **({'control_head_rows': backbone.control_head_rows.detach().cpu()} if not getattr(backbone, 'tied', True) else {}), 'lora': lora_state(backbone),
            'lora_layers': adapter_layers(backbone), 'lora_rank': next(iter(ranks), 0),
            'backbone_training': args.backbone_training,
            **({'maple_qat': True} if qat_named else {}),
            **({'backbone_trainables': policy_state()} if args.backbone_training in ('full','qat') else {}),
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
