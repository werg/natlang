#!/usr/bin/env python3
"""Supervised fine-tuning of the interpreter on exported pairs (scripts/export_sft.py), completion-only loss.

LoRA adapters on a bf16 base, optional gradient checkpointing, length-aware microbatches,
completion-only loss, and optional persistent token caching. --accum counts sequences per optimizer step. `--full` trains all
weights instead (needs far more memory).

Every run is stoppable and resumable. A checkpoint (adapter or full weights, optimizer, scheduler, step, position
in the data, random state) is written every `--save-every` optimizer steps and when the process is asked to stop
(SIGTERM or Ctrl-C: it finishes the current step first). Starting the same command again continues from the
checkpoint; the data order is a function of `--seed`, so a resumed run sees exactly the pairs it would have seen.
A servable model can be exported from the latest checkpoint at any time, also while no training is running:

  docker run --rm --gpus all -v "$PWD:/work" -e HF_HOME=/work/models/hf --name natlang-train natlang-train \\
    python scripts/train_lora.py data/sft.jsonl runs/lora --steps 600          # start, or resume
  docker stop -t 120 natlang-train                                             # stop cleanly (a checkpoint is written)
  ... python scripts/train_lora.py data/sft.jsonl runs/lora --merge-only       # runs/lora/merged from the checkpoint
"""
import argparse, json, math, os, random, shutil, signal, time, sys, sqlite3
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.corpus import split_programs, file_digest, digest, index_pairs

import torch


def restore_lfm_expert_quantization(model, model_dir):
    """Attach the saved NF4 state to LFM's raw MoE expert tensors.

    BitsAndBytes discovers ordinary Linear modules itself.  LFM experts are
    packed tensors, so a prequantized checkpoint carries their QuantState in a
    companion file and needs this one-time restoration before the first
    forward pass.
    """
    from bitsandbytes.nn import Params4bit
    from safetensors import safe_open

    model_dir = Path(model_dir)
    states_path = model_dir / "expert_quant_state.pt"
    weights_path = model_dir / "model.safetensors"
    if not states_path.exists() or not weights_path.exists():
        raise FileNotFoundError("LFM expert QLoRA requires model.safetensors and expert_quant_state.pt")
    states = torch.load(states_path, weights_only=False)
    device = next(model.parameters()).device
    for state in states.values():
        state.absmax = state.absmax.to(device, non_blocking=True)
        state.code = state.code.to(device, non_blocking=True)
        if state.nested:
            if state.offset is not None:
                state.offset = state.offset.to(device, non_blocking=True)
            if getattr(state.state2, "absmax", None) is not None:
                state.state2.absmax = state.state2.absmax.to(device, non_blocking=True)
            if getattr(state.state2, "code", None) is not None:
                state.state2.code = state.state2.code.to(device, non_blocking=True)
    restored = 0
    with safe_open(str(weights_path), framework="pt", device="cpu") as weights:
        for name, param in list(model.named_parameters()):
            if name not in states or "experts" not in name:
                continue
            packed = Params4bit(param.data, requires_grad=False, quant_state=states[name],
                                blocksize=64, compress_statistics=True, quant_type="nf4",
                                bnb_quantized=True)
            packed.data.copy_(weights.get_tensor(name).to(device))
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], packed)
            restored += 1
    if not restored:
        raise ValueError("checkpoint contained no restorable LFM expert weights")
    print(f"restored NF4 state for {restored} LFM expert tensors", flush=True)


def directory_digest(path):
    """Content identity for an adapter used to initialize a new phase."""
    path = Path(path)
    h = __import__("hashlib").sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(item.relative_to(path)).encode())
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def completion_loss(model, encoded):
    """Keep the preceding prompt position so the first completion token is trained.

    The transformer still reads the whole input. Only the vocabulary projection
    and cross-entropy for ignored prompt positions are omitted.
    """
    inputs, labels = encoded
    return model(input_ids=inputs, labels=labels, logits_to_keep=labels.shape[-1]).loss



def collate_completions(examples, pad_id=0, device="cuda"):
    """Right padding preserves causal/convolution history; prompts never carry loss."""
    length = max(len(x) + len(y) for x, y in examples)
    start = min(len(x) for x, y in examples) - 1
    inputs = torch.full((len(examples), length), pad_id, dtype=torch.long)
    mask = torch.zeros_like(inputs)
    labels = torch.full_like(inputs, -100)
    for i, (x, y) in enumerate(examples):
        end = len(x) + len(y)
        inputs[i, :end] = torch.tensor(x + y)
        mask[i, :end] = 1
        labels[i, len(x):end] = torch.tensor(y)
    return {"input_ids": inputs.to(device), "attention_mask": mask.to(device),
            "labels": labels[:, start:].to(device)}


def batch_completion_loss(model, encoded):
    """Mean of per-example completion losses, matching single-example accumulation."""
    if encoded["input_ids"].shape[0] == 1:
        # Let Transformers use the model's fused causal-loss path.  The manual
        # reduction below is needed only to give every item in a padded batch
        # equal weight regardless of completion length.
        return completion_loss(model, (encoded["input_ids"], encoded["labels"]))
    labels = encoded["labels"][:, 1:]
    logits = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                   logits_to_keep=encoded["labels"].shape[-1]).logits[:, :-1].float()
    losses = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                               labels.reshape(-1), ignore_index=-100, reduction="none")
    losses = losses.reshape_as(labels)
    return (losses.sum(dim=1) / (labels != -100).sum(dim=1)).mean()


def microbatches(examples, size, token_budget):
    """Only group within one optimizer step; no examples cross a split or step."""
    batch = []
    for example in sorted(examples, key=lambda e: len(e[0]) + len(e[1])):
        length = len(example[0]) + len(example[1])
        if batch and (len(batch) >= size or length * (len(batch) + 1) > token_budget):
            yield batch
            batch = []
        batch.append(example)
    if batch:
        yield batch


def set_layer_checkpointing(model, enabled, retain_every_n_layers=0):
    """Checkpoint a configurable subset of Transformers decoder layers.

    Transformers 5.5 only exposes all-or-nothing checkpointing, although each
    GradientCheckpointingLayer has an independent flag.  Keeping activations
    for a sparse set of layers trades otherwise idle VRAM for less backward
    recomputation.  A value of four retains every fourth layer and checkpoints
    the other three.
    """
    if enabled:
        model.gradient_checkpointing_enable()
        from transformers.modeling_layers import GradientCheckpointingLayer
        layers = [module for module in model.modules()
                  if isinstance(module, GradientCheckpointingLayer)]
        if retain_every_n_layers:
            for index, layer in enumerate(layers):
                layer.gradient_checkpointing = index % retain_every_n_layers != 0
        return len(layers), sum(bool(layer.gradient_checkpointing) for layer in layers)
    model.gradient_checkpointing_disable()
    return 0, 0


class TokenCache:
    """Lazy persistent token cache, bound to source bytes and tokenizer behavior."""
    def __init__(self, path, identity):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (identity TEXT)")
        stored = self.db.execute("SELECT identity FROM metadata").fetchone()
        key = json.dumps(identity, sort_keys=True)
        if stored and stored[0] != key:
            self.db.close()
            raise ValueError("Token cache source/tokenizer mismatch; choose a different --token-cache")
        if not stored:
            self.db.execute("INSERT INTO metadata VALUES (?)", (key,))
        self.db.execute("CREATE TABLE IF NOT EXISTS tokens (offset INTEGER PRIMARY KEY, data TEXT)")
        self.db.commit()

    def get(self, offset):
        row = self.db.execute("SELECT data FROM tokens WHERE offset=?", (offset,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, offset, value):
        self.db.execute("INSERT OR REPLACE INTO tokens VALUES (?, ?)", (offset, json.dumps(value)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--model-revision", help="immutable Hugging Face commit or tag for the base model")
    schedule = ap.add_mutually_exclusive_group()
    schedule.add_argument("--steps", type=int, help="optimizer steps in total (default: 300)")
    schedule.add_argument("--epochs", type=float,
                          help="corpus passes; each admitted training example is used once per epoch")
    ap.add_argument("--accum", type=int, default=16, help="sequences per optimizer step (unchanged by microbatch size)")
    ap.add_argument("--microbatch", type=int, default=1)
    ap.add_argument("--batch-tokens", type=int, default=8192, help="maximum padded tokens per microbatch; long examples run alone")
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--checkpoint-above-tokens", type=int, default=0, help="with checkpointing enabled, skip recomputation for microbatches at or below this padded-token count")
    ap.add_argument("--retain-every-n-layers", type=int, default=0,
                    help="retain activations for every Nth decoder layer; zero checkpoints every layer")
    ap.add_argument("--token-cache", type=Path)
    ap.add_argument("--benchmark-steps", type=int, default=0, help="isolated throughput run, no heldout evaluation or saved model")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="QLoRA base loading for models that do not fit in bf16 (checkpoint stores the adapter)")
    ap.add_argument("--unsloth", action="store_true",
                    help="load and patch a dense model through Unsloth")
    ap.add_argument("--unsloth-lfm-experts", action="store_true",
                    help="load an LFM NF4 checkpoint with packed MoE experts through Unsloth")
    ap.add_argument("--unsloth-compile", action="store_true",
                    help="enable Unsloth torch.compile paths (off by default for BitsAndBytes compatibility)")
    ap.add_argument("--target-modules",
                    help="comma-separated LoRA module suffixes; LFM uses its architecture-specific linear layers")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=200, help="minimum turns held out, reserving whole programs")
    ap.add_argument("--save-every", type=int, default=25, help="checkpoint every N optimizer steps")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="also retain adapter-only snapshots every N steps for behavioral selection")
    ap.add_argument("--init-adapter", type=Path,
                    help="start a new training phase from this LoRA adapter with a fresh optimizer")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start over")
    ap.add_argument("--merge-only", action="store_true", help="export out/merged from the latest checkpoint and exit")
    a = ap.parse_args()
    if a.steps is not None and a.steps < 1:
        ap.error("--steps must be positive")
    if a.epochs is not None and a.epochs <= 0:
        ap.error("--epochs must be positive")
    if min(a.accum, a.microbatch, a.batch_tokens, a.save_every) < 1 or min(a.benchmark_steps, a.checkpoint_above_tokens, a.snapshot_every, a.retain_every_n_layers) < 0:
        ap.error("batch sizes and save interval must be positive; benchmark steps nonnegative")
    if not a.gradient_checkpointing and a.retain_every_n_layers:
        ap.error("--retain-every-n-layers requires gradient checkpointing")
    if a.full and a.load_in_4bit:
        ap.error("4-bit loading is for LoRA adapters, not full-weight training")
    if a.unsloth_lfm_experts and not a.load_in_4bit:
        ap.error("--unsloth-lfm-experts requires --load-in-4bit")
    if a.unsloth and not a.load_in_4bit:
        ap.error("--unsloth requires --load-in-4bit in this memory-constrained trainer")
    if a.merge_only and a.load_in_4bit:
        ap.error("a QLoRA checkpoint is adapter-only; convert the adapter or merge it with a non-quantized base")
    if a.init_adapter is not None and not a.init_adapter.is_dir():
        ap.error("--init-adapter must be an adapter directory")
    targets = [x.strip() for x in a.target_modules.split(",") if x.strip()] if a.target_modules else None
    if a.target_modules and not targets:
        ap.error("--target-modules must name at least one module")
    default_targets = (["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]
                       if a.unsloth_lfm_experts else ["q_proj", "k_proj", "v_proj", "o_proj"])
    ckpt = a.out / "checkpoint"
    state_file = ckpt / "state.json"
    if a.fresh and ckpt.exists():
        shutil.rmtree(ckpt)
    resume = state_file.exists()
    state = json.loads(state_file.read_text()) if resume else {"step": 0, "cursor": 0, "skipped": 0, "log": []}
    if a.benchmark_steps and (resume or a.merge_only):
        ap.error("benchmarks require a separate output directory without a checkpoint")
    if a.merge_only and not resume:
        raise SystemExit(f"no checkpoint in {ckpt}")

    if not a.merge_only:
        pairs = index_pairs(a.data)
        held, train, split = split_programs(pairs, a.holdout, a.seed)
        target_examples = (max(1, math.ceil(len(train) * a.epochs))
                           if a.epochs is not None else (a.steps or 300) * a.accum)
        a.steps = math.ceil(target_examples / a.accum)
        identity = {"data_sha256": file_digest(a.data), "split_sha256": digest(split),
                    "max_len": a.max_len, "model": a.model,
                    "model_revision": a.model_revision, "accum": a.accum,
                    "target_examples": target_examples, "steps": a.steps, "lr": a.lr,
                    "gradient_checkpointing": a.gradient_checkpointing,
                    "checkpoint_above_tokens": a.checkpoint_above_tokens}
        if a.retain_every_n_layers:
            identity["retain_every_n_layers"] = a.retain_every_n_layers
        if a.init_adapter is not None:
            identity["init_adapter"] = {"path": str(a.init_adapter),
                                        "sha256": directory_digest(a.init_adapter)}
        if a.load_in_4bit:
            identity["qlora"] = {"load_in_4bit": True, "rank": a.rank, "quant_type": "nf4",
                                 "double_quant": True,
                                 "target_modules": targets or default_targets,
                                 "unsloth_lfm_experts": a.unsloth_lfm_experts,
                                 "unsloth_compile": a.unsloth_compile}
            if a.unsloth and not a.unsloth_lfm_experts:
                identity["qlora"]["unsloth"] = True
        if resume and state.get("corpus") != identity:
            raise SystemExit("Checkpoint corpus/split/settings differ (or predate program splits). "
                             "Use --merge-only to export it, or a new output directory for this training run.")
        state["corpus"] = identity
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "split.json").write_text(json.dumps(split, indent=2) + "\n")
        print(f"program split: {len(train)} training turns, {len(held)} held-out turns "
              f"from {len(split['held_programs'])} programs; target {target_examples} examples "
              f"({target_examples / len(train):.3g} epochs, {a.steps} optimizer steps)", flush=True)
    torch.manual_seed(a.seed)
    FastLanguageModel = None
    use_unsloth = a.unsloth or a.unsloth_lfm_experts
    if use_unsloth:  # Unsloth must patch Transformers before PEFT imports it.
        compile_root = Path(os.environ.get("HF_HOME", a.out / "hf-cache")) / "unsloth_compiled_cache"
        compile_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("UNSLOTH_COMPILE_LOCATION", str(compile_root))
        if not a.unsloth_compile:
            os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
            os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        from unsloth import FastLanguageModel
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base_src = str(ckpt / "weights") if (resume and a.full) else a.model
    if use_unsloth:
        from huggingface_hub import snapshot_download
        model_path = Path(a.model)
        base_src = (str(model_path.resolve()) if model_path.is_dir() else
                    snapshot_download(a.model, revision=a.model_revision))
        model, tok = FastLanguageModel.from_pretrained(
            model_name=base_src, max_seq_length=a.max_len, load_in_4bit=True, device_map=0)
        if a.unsloth_lfm_experts:
            restore_lfm_expert_quantization(model, base_src)
    else:
        tok = AutoTokenizer.from_pretrained(a.model, revision=a.model_revision)
        load_options = {"dtype": torch.bfloat16}
        if a.model_revision:
            load_options["revision"] = a.model_revision
        if a.load_in_4bit:
            load_options.update({"device_map": {"": 0}, "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)})
        model = AutoModelForCausalLM.from_pretrained(base_src, **load_options)
        if not a.load_in_4bit:
            model = model.cuda()
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if not a.full:
        if a.load_in_4bit and not a.unsloth_lfm_experts:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=a.gradient_checkpointing)
        if resume:
            model = PeftModel.from_pretrained(model, str(ckpt / "weights"), is_trainable=True)
        elif a.init_adapter is not None:
            model = PeftModel.from_pretrained(model, str(a.init_adapter), is_trainable=True)
        else:
            linear = targets
            if linear is None:
                linear = (default_targets if a.load_in_4bit else
                          sorted({n.split(".")[-1] for n, m in model.named_modules()
                                  if isinstance(m, torch.nn.Linear) and "lm_head" not in n}))
            if use_unsloth:
                model = FastLanguageModel.get_peft_model(
                    model, r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                    target_modules=linear, use_gradient_checkpointing="unsloth", random_state=a.seed)
            else:
                model = get_peft_model(model, LoraConfig(
                    r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                    target_modules=linear, task_type="CAUSAL_LM"))
        model.enable_input_require_grads()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)", flush=True)

    if a.gradient_checkpointing:
        layer_count, checkpointed_count = set_layer_checkpointing(
            model, True, a.retain_every_n_layers)
        print(f"activation checkpointing: {checkpointed_count}/{layer_count} decoder layers", flush=True)

    def export_merged():
        merged = model.merge_and_unload() if not a.full else model
        merged.save_pretrained(a.out / "merged", safe_serialization=True)
        tok.save_pretrained(a.out / "merged")
        (a.out / "merged" / "natlang_training.json").write_text(json.dumps({k: state[k] for k in ("step", "cursor")} | {"data": str(a.data), "corpus": state.get("corpus")}))
        print(f"saved {a.out / 'merged'} (step {state['step']})", flush=True)

    if a.merge_only:
        export_merged()
        return


    data_stream = a.data.open("rb")
    cache = TokenCache(a.token_cache, {"source": state["corpus"]["data_sha256"],
                                      "tokenizer": digest(tok.backend_tokenizer.to_str())}) if a.token_cache else None

    def encode(p, phase="train"):        # already rendered by the chat template: no special tokens added
        offset = p["offset"]
        cached = cache.get(offset) if cache else None
        if cached is None:
            data_stream.seek(offset)
            p = json.loads(data_stream.readline())
            x = tok(p["prompt"], add_special_tokens=False)["input_ids"]
            y = tok(p["completion"], add_special_tokens=False)["input_ids"]
            family = p.get("family", "unknown")
            if cache:
                cache.put(offset, [x, y, family])
        else:
            x, y, family = cached
        if not x or not y:
            raise ValueError("Training pairs require a nonempty prompt and completion")
        if len(x) + len(y) > a.max_len:
            counts = state.setdefault("overlength_encounters", {}).setdefault(phase, {})
            counts[family] = counts.get(family, 0) + 1
            return None
        return x, y

    @torch.no_grad()
    def heldout_loss():
        model.eval()
        tot = n = 0
        for p in held[:100]:
            e = encode(p, phase="heldout")
            if e:
                tot += batch_completion_loss(model, collate_completions([e])).item(); n += 1
        model.train()
        if held and not n:
            raise ValueError("Every held-out example exceeds --max-len")
        return tot / n if n else None

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    if resume:
        opt.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location="cuda"))
        sched.load_state_dict(torch.load(ckpt / "scheduler.pt"))
        torch.set_rng_state(torch.load(ckpt / "rng.pt"))
        print(f"resumed from step {state['step']} (pair {state['cursor']})", flush=True)
    else:
        torch.manual_seed(a.seed)
        state["heldout_before"] = None if a.benchmark_steps else heldout_loss()
        print(f"{len(train)} training pairs; held-out loss before: {state['heldout_before']}", flush=True)

    def save_checkpoint():
        tmp = a.out / "checkpoint.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        model.save_pretrained(tmp / "weights", safe_serialization=True)
        torch.save(opt.state_dict(), tmp / "optimizer.pt")
        torch.save(sched.state_dict(), tmp / "scheduler.pt")
        torch.save(torch.get_rng_state(), tmp / "rng.pt")
        (tmp / "state.json").write_text(json.dumps({**state, "args": {k: str(v) for k, v in vars(a).items()}}))
        if ckpt.exists():                                  # swap in atomically enough: the old one goes last
            old = a.out / "checkpoint.old"
            if old.exists():
                shutil.rmtree(old)
            os.rename(ckpt, old)
            os.rename(tmp, ckpt)
            shutil.rmtree(old)
        else:
            os.rename(tmp, ckpt)
        if a.snapshot_every and state["step"] % a.snapshot_every == 0:
            snapshot = a.out / "snapshots" / f"step-{state['step']:04d}"
            snapshot_tmp = snapshot.with_name(snapshot.name + ".tmp")
            if snapshot_tmp.exists():
                shutil.rmtree(snapshot_tmp)
            snapshot_tmp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(ckpt / "weights", snapshot_tmp / "weights")
            shutil.copy2(ckpt / "state.json", snapshot_tmp / "state.json")
            if snapshot.exists():
                shutil.rmtree(snapshot)
            os.rename(snapshot_tmp, snapshot)

    stop = {"now": False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.update(now=True))
    a.out.mkdir(parents=True, exist_ok=True)
    model.train()
    torch.cuda.synchronize()
    t0 = time.time()
    metrics_file = a.out / "throughput.json"
    metrics = ([m for m in json.loads(metrics_file.read_text())["steps"] if m["step"] <= state["step"]]
               if resume and metrics_file.exists() else [])
    packages = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "unsloth",
                 "unsloth_zoo", "causal-conv1d"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    def save_metrics():
        tmp = metrics_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()}, "corpus": state["corpus"],
            "packages": packages, "gpu": torch.cuda.get_device_name(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "steps": metrics}, indent=2) + "\n")
        tmp.replace(metrics_file)

    target_steps = a.benchmark_steps or a.steps
    target_examples = a.benchmark_steps * a.accum if a.benchmark_steps else state["corpus"]["target_examples"]
    state.setdefault("trained_examples", state["step"] * a.accum)
    while state["step"] < target_steps and state["trained_examples"] < target_examples and not stop["now"]:
        step_start = time.perf_counter()
        examples = []
        for _ in range(min(a.accum, target_examples - state["trained_examples"])):
            e = None
            start_cursor = state["cursor"]
            while e is None:
                if state["cursor"] - start_cursor >= len(train):
                    raise ValueError("Every training example exceeds --max-len")
                e = encode(train[state["cursor"] % len(train)]); state["cursor"] += 1
                state["skipped"] += e is None
            examples.append(e)
        ready = time.perf_counter()
        running = torch.zeros((), device="cuda")
        batches = 0
        padded_tokens = 0
        for batch in microbatches(examples, a.microbatch, a.batch_tokens):
            encoded = collate_completions(batch, pad_id=tok.pad_token_id or 0)
            want_checkpointing = a.gradient_checkpointing and encoded["input_ids"].numel() > a.checkpoint_above_tokens
            if want_checkpointing != model.is_gradient_checkpointing:
                set_layer_checkpointing(model, want_checkpointing, a.retain_every_n_layers)
            loss = batch_completion_loss(model, encoded) * (len(batch) / len(examples))
            loss.backward()
            running += loss.detach()
            padded_tokens += encoded["input_ids"].numel()
            batches += 1
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        state["step"] += 1
        state["trained_examples"] += len(examples)
        running = running.item()  # one synchronization per optimizer step, not per sequence
        if cache:
            cache.db.commit()
        metrics.append({"step": state["step"], "seconds": time.perf_counter() - step_start,
                        "prepare_seconds": ready - step_start, "examples": len(examples),
                        "tokens": sum(len(x) + len(y) for x, y in examples),
                        "completion_tokens": sum(len(y) for x, y in examples),
                        "padded_tokens": padded_tokens, "microbatches": batches, "loss": running})
        if state["step"] % 10 == 0 or state["step"] == target_steps:
            save_metrics()
            state["log"].append([state["step"], round(running, 4)])
            print(f"step {state['step']:4d}  loss {running:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB  {time.time() - t0:.0f}s "
                  f"overlength={state.get('overlength_encounters', {})}", flush=True)
        if not a.benchmark_steps and state["step"] % a.save_every == 0:
            save_checkpoint()
    save_metrics()
    if a.benchmark_steps:
        if cache:
            cache.db.close()
        data_stream.close()
        return
    save_checkpoint()
    if stop["now"]:
        action = ("the checkpoint adapter can be converted or served directly"
                  if a.load_in_4bit else "use --merge-only to export this state")
        print(f"stopped at step {state['step']} of {a.steps}; checkpoint written. "
              f"Run the same command to continue, or {action}.", flush=True)
        return
    state["heldout_after"] = heldout_loss()
    print(f"held-out loss after: {state['heldout_after']}; trained examples {state['trained_examples']}, "
          f"source rows visited {state['cursor']}, too long {state['skipped']}", flush=True)
    save_checkpoint()
    if a.load_in_4bit:
        print(f"saved adapter {ckpt / 'weights'}; QLoRA checkpoints are not merged into their quantized base", flush=True)
    else:
        export_merged()
    if cache:
        cache.db.commit()
        cache.db.close()
    data_stream.close()


if __name__ == "__main__":
    main()
