#!/usr/bin/env python3
"""Supervised fine-tuning of the interpreter on exported pairs (ts-host/scripts/export-native-sft.mjs), completion-only loss.

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
from scripts.corpus import split_programs, file_digest, digest, index_pairs
from scripts.training_append import append_train_order, cosine_extension_multiplier, extended_target_examples
from scripts.training_exclusion import (exclusion_split_manifest, exclusion_target,
                                        filter_training_order, pending_source_review_entry,
                                        post_exclusion_mix_summary,
                                        validate_exclusion_checkpoint_resume)
from scripts.training_readiness import (clip_finite_grad_norm_, require_finite_loss,
                                        validate_training_audit,
                                        validate_training_audit_tokenizer,
                                        validate_training_inventory_audit,
                                        validate_training_mix_audit)
from scripts.periodic_heldout import (append_periodic_metric, canonical_sha256,
                                      evaluate_fixed_subset, periodic_metric_already_complete,
                                      hydrate_support_metadata, read_periodic_metrics, select_fixed_subset,
                                      trainable_weights_sha256)

import torch


# Routed experts can be stacked parameters ("…experts.gate_up_proj") or
# individual linear modules ("…experts.12.gate_proj", as in Ling).
# Shared experts keep the ordinary adapter rank.
EXPERTS = r"experts(?:\.\d+)?(?:\.\w+)?"


def split_targets(model, targets):
    """Target names as PEFT takes them: linear layers by leaf name, and stacked expert weights (a MoE's experts
    module holding one parameter for all experts) as "module.parameter"; a name can be both."""
    import torch.nn as nn
    layers, stacked = set(), set()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(module, nn.Linear) and leaf in targets:
            layers.add(leaf)
        for target in targets:
            weight = module._parameters.get(target)
            if weight is not None and (weight.dim() == 3 or getattr(weight, "_original_shape", None) is not None):
                stacked.add(f"{leaf}.{target}")
    return sorted(layers), sorted(stacked)


def require_adapter_coverage(model, targets, exclude=None):
    """Fail unless every linear layer a target names, and `exclude` does not, received an adapter; report how many
    each target got."""
    import re
    import torch.nn as nn
    wrapped, missed = {}, {}
    for name, module in model.named_modules():
        parameter = getattr(module, "parameter_name", None)
        if parameter in targets and hasattr(module, "lora_A"):
            # LoRA on a raw parameter, such as a MoE's stacked experts; two on one module nest via base_layer.
            wrapped[parameter] = wrapped.get(parameter, 0) + 1
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets or ".lora_" in name or ".base_layer" in name:
            continue
        if exclude and re.fullmatch(exclude, name):
            continue
        if hasattr(module, "lora_A"):
            wrapped[leaf] = wrapped.get(leaf, 0) + 1
        elif isinstance(module, nn.Linear):
            missed.setdefault(leaf, []).append(name)
    print("adapters per target: " + ", ".join(f"{t}={wrapped.get(t, 0)}" for t in targets), flush=True)
    if missed:
        raise RuntimeError("target modules left without an adapter: " +
                           "; ".join(f"{leaf} ({len(names)}, e.g. {names[0]})" for leaf, names in missed.items()))
    absent = [t for t in targets if not wrapped.get(t)]
    if absent:
        raise RuntimeError(f"target modules match no layer: {', '.join(absent)}")


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
            # The logical (experts, out, in) shape, which LoRA on the stacked experts needs to see past the packing.
            packed._original_shape = torch.Size(states[name].shape)
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


def recover_checkpoint_directory(out):
    """Recover the last complete checkpoint after interruption during directory swap."""
    out = Path(out)
    checkpoint, old, temporary = (out / name for name in
                                  ("checkpoint", "checkpoint.old", "checkpoint.tmp"))
    if not checkpoint.exists() and old.exists():
        os.replace(old, checkpoint)
        fsync_directory(out)
    if checkpoint.exists():
        if old.exists():
            shutil.rmtree(old)
        if temporary.exists():
            shutil.rmtree(temporary)
        fsync_directory(out)


def fsync_file(path):
    """Flush a completed checkpoint file before publishing its directory."""
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())


def fsync_directory(path):
    """Persist directory entries across power loss on filesystems supporting fsync."""
    fd = os.open(Path(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_checkpoint_directory(out, write_weights, optimizer_state, scheduler_state,
                              rng_state, state_payload):
    """Install a complete checkpoint directory with recoverable directory swaps.

    ``write_weights`` writes the model-specific files into ``tmp/weights``;
    optimizer, scheduler, RNG, and state serialization is shared with CPU
    recovery tests so the tested checkpoint boundary is the production one.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "checkpoint.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        write_weights(tmp / "weights")
        torch.save(optimizer_state, tmp / "optimizer.pt")
        torch.save(scheduler_state, tmp / "scheduler.pt")
        torch.save(rng_state, tmp / "rng.pt")
        (tmp / "state.json").write_text(json.dumps(state_payload))
        for item in sorted(p for p in tmp.rglob("*") if p.is_file()):
            fsync_file(item)
        for directory in sorted((p for p in tmp.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            fsync_directory(directory)
        fsync_directory(tmp)
        checkpoint, old = out / "checkpoint", out / "checkpoint.old"
        if checkpoint.exists():
            if old.exists():
                shutil.rmtree(old)
            os.rename(checkpoint, old)
            fsync_directory(out)
            os.rename(tmp, checkpoint)
            fsync_directory(out)
            shutil.rmtree(old)
            fsync_directory(out)
        else:
            os.rename(tmp, checkpoint)
            fsync_directory(out)
    except BaseException:
        # Preserve a previous complete directory if promotion was interrupted.
        recover_checkpoint_directory(out)
        raise


def install_stop_handlers():
    """Install deferred stop handlers early enough to cover loading and evaluation."""
    stop = {"now": False}
    def request_stop(*_):
        stop["now"] = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    return stop


def capture_rng_state():
    return {"torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate()}


def restore_rng_state(saved):
    # Accept checkpoints from the previous CPU-only RNG format.
    if isinstance(saved, torch.Tensor):
        torch.set_rng_state(saved)
        return
    torch.set_rng_state(saved["torch_cpu"])
    if saved.get("torch_cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["torch_cuda"])
    if saved.get("python") is not None:
        random.setstate(saved["python"])


def order_training_pairs(train, data_order):
    if data_order == "source":
        train.sort(key=lambda row: row["offset"])
    elif data_order != "shuffle":
        raise ValueError(f"Unsupported training data order {data_order!r}")
    return train


def validate_heldout_rows(heldout_rows, split):
    """Fail closed if gate validation accidentally shadows the held-out rows."""
    if not isinstance(heldout_rows, list):
        raise ValueError("held-out rows were overwritten during training-gate validation")
    expected = split.get("held_turns") if isinstance(split, dict) else None
    if not isinstance(expected, int) or len(heldout_rows) != expected:
        raise ValueError("held-out row count differs from the preserved split")
    if any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in heldout_rows):
        raise ValueError("held-out rows were overwritten during training-gate validation")


def completion_loss(model, encoded):
    """Keep the preceding prompt position so the first completion token is trained.

    The transformer still reads the whole input. Only the vocabulary projection
    and cross-entropy for ignored prompt positions are omitted.
    """
    inputs, labels = encoded
    return model(input_ids=inputs, labels=labels, logits_to_keep=labels.shape[-1]).loss



def collate_completions(examples, pad_id=0, device="cuda"):
    """Right padding preserves causal/convolution history; prompts never carry loss.

    An example is (prompt, completion) or, for a merged conversation, (prompt, rest, trained): rest holds each later
    turn's context and completion, and trained marks which of its tokens carry loss."""
    length = max(len(e[0]) + len(e[1]) for e in examples)
    start = min(len(e[0]) for e in examples) - 1
    inputs = torch.full((len(examples), length), pad_id, dtype=torch.long)
    mask = torch.zeros_like(inputs)
    labels = torch.full_like(inputs, -100)
    for i, e in enumerate(examples):
        x, y = e[0], e[1]
        end = len(x) + len(y)
        inputs[i, :end] = torch.tensor(x + y)
        mask[i, :end] = 1
        labels[i, len(x):end] = torch.tensor(y if len(e) == 2 else [t if keep else -100 for t, keep in zip(y, e[2])])
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
        from transformers.modeling_layers import GradientCheckpointingLayer
        layers = [module for module in model.modules()
                  if isinstance(module, GradientCheckpointingLayer)]
        if retain_every_n_layers and not layers:
            raise ValueError(
                "Selective activation checkpointing requires per-layer "
                "GradientCheckpointingLayer support; this model uses a different "
                "checkpointing implementation. Use --retain-every-n-layers 0 "
                "for its native full checkpointing."
            )
        model.gradient_checkpointing_enable()
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
    ap.add_argument("--repair-mean-nll", type=float, help="enable online corrective-prefix SFT above this mean target NLL")
    ap.add_argument("--repair-full-gold-every", type=int, default=10, help="retain full gold supervision every N examples in repair mode; zero disables")
    ap.add_argument("--repair-token-nll", type=float, help="first token above this NLL triggers complete-action prefix supervision")
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="LiquidAI/LFM2.5-350M")
    ap.add_argument("--model-revision", help="immutable Hugging Face commit or tag for the base model")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                    help="execution device; CPU is useful for small pipeline smoke runs")
    ap.add_argument("--cuda-memory-fraction", type=float, default=1.0,
                    help="fraction of CUDA device memory available to this training process")
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
    ap.add_argument("--profile", action="store_true",
                    help="with --benchmark-steps N >= 2, profile the last step and print the operators by GPU time")
    ap.add_argument("--max-len", type=int, default=3072)
    ap.add_argument("--require-audit", action="store_true",
                    help="require a ready sibling .manifest.json matching the data bytes and --max-len")
    ap.add_argument("--require-mix-audit", type=Path,
                    help="require a version-2 reducer mix report bound to this exact ready corpus")
    ap.add_argument("--required-reducer-share", type=float, default=0.25,
                    help="required reducer share when --require-mix-audit is set")
    ap.add_argument("--require-data-inventory-ready", type=Path,
                    help="require an inventory readiness record bound to an immutable report")
    ap.add_argument("--inventory-policy", type=Path,
                    help="current data-source policy whose hash must match the inventory report")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="QLoRA base loading for models that do not fit in bf16 (checkpoint stores the adapter)")
    ap.add_argument("--trust-remote-code", action="store_true",
                    help="load a model whose repository ships its own modeling code (for example Ling-3.0's bailing_hybrid)")
    ap.add_argument("--unsloth", action="store_true",
                    help="load and patch a dense model through Unsloth")
    ap.add_argument("--unsloth-lfm-experts", action="store_true",
                    help="load an LFM NF4 checkpoint with packed MoE experts through Unsloth")
    ap.add_argument("--unsloth-compile", action="store_true",
                    help="enable Unsloth torch.compile paths (off by default for BitsAndBytes compatibility)")
    ap.add_argument("--target-modules",
                    help="comma-separated LoRA module suffixes; LFM uses its architecture-specific linear layers")
    ap.add_argument("--optimizer", choices=("adamw", "paged-adamw-8bit", "muon"), default="adamw",
                    help="paged-adamw-8bit keeps optimizer state in 8 bits and pages it to CPU memory under pressure")
    ap.add_argument("--muon-momentum", type=float, default=0.95)
    ap.add_argument("--muon-ns-steps", type=int, default=5)
    ap.add_argument("--unsloth-moe", action="store_true",
                    help="load through Transformers but with Unsloth's MoE support: stacked experts quantized to 4 "
                         "bits, run as one grouped matmul, and adaptable by LoRA (for models Unsloth cannot load)")
    ap.add_argument("--expert-rank", type=int,
                    help="LoRA rank for routed MoE experts, stacked or individual (default: --rank); shared experts "
                         "keep --rank")
    ap.add_argument("--moe-backend", choices=("grouped_mm", "unsloth_triton", "native_torch"),
                    help="with --unsloth-moe, the expert matmul backend (default: Unsloth's choice, grouped_mm when "
                         "torch has it; it loops over experts below sm90, where unsloth_triton runs one kernel)")
    ap.add_argument("--offload-checkpoints", action=argparse.BooleanOptionalAction, default=True,
                    help="with --unsloth-moe, keep checkpointed layer inputs in CPU memory (Unsloth's smart "
                         "checkpointing); --no-offload-checkpoints keeps them on the GPU when they fit")
    ap.add_argument("--no-kbit-upcast", action="store_true",
                    help="with --load-in-4bit, skip PEFT's k-bit preparation, which upcasts every unquantized weight "
                         "(a large vocabulary's embedding among them) to fp32; train in bf16 as Unsloth does")
    ap.add_argument("--no-expert-lora", action="store_true",
                    help="adapt no stacked MoE expert weights (attention, shared experts and dense layers only): "
                         "faster, since routed-expert LoRA runs per expert when its rank misaligns grouped matmuls, "
                         "and smaller")
    ap.add_argument("--exclude-modules",
                    help="regex over full module names kept out of the adapter, e.g. a MoE's routed experts")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=200, help="minimum turns held out, reserving whole programs")
    ap.add_argument("--skip-heldout-loss", action="store_true",
                    help="skip before/after held-out loss evaluation while retaining the held-out split")
    ap.add_argument("--periodic-heldout-every", type=int, default=0,
                    help="evaluate a fixed deterministic held-out subset every N optimizer steps; zero disables it")
    ap.add_argument("--periodic-heldout-count", type=int, default=128,
                    help="maximum number of held-out rows in the periodic subset")
    ap.add_argument("--periodic-heldout-at-start", action="store_true",
                    help="on resume, evaluate the current checkpoint once even if it is not an interval multiple")
    ap.add_argument("--data-order", choices=("shuffle", "source"), default="shuffle",
                    help="shuffle training rows (default) or preserve source file order")
    ap.add_argument("--save-every", type=int, default=25, help="checkpoint every N optimizer steps")
    ap.add_argument("--snapshot-every", type=int, default=0,
                    help="also retain adapter-only snapshots every N steps for behavioral selection")
    ap.add_argument("--init-adapter", type=Path,
                    help="start a new training phase from this LoRA adapter with a fresh optimizer")
    ap.add_argument("--append-manifest", type=Path,
                    help="authorize a source-safe append transition from the exact checkpoint copied into this output")
    ap.add_argument("--exclusion-manifest", type=Path,
                    help="resume a copied checkpoint after a reviewed row exclusion while preserving its original split/order")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start over")
    ap.add_argument("--merge-only", action="store_true", help="export out/merged from the latest checkpoint and exit")
    ap.add_argument("--no-merge", action="store_true",
                    help="save the resumable checkpoint but skip exporting a merged model at completion")
    ap.add_argument("--phase-manifest", type=Path,
                    help="start/resume a reviewed post-training phase from an exact complete copied checkpoint")
    a = ap.parse_args()
    if a.phase_manifest and (a.append_manifest or a.exclusion_manifest or a.fresh or a.init_adapter or a.merge_only or a.benchmark_steps):
        ap.error("post-training phase requires a copied checkpoint and no concurrent transition/reset")
    if a.phase_manifest and not (a.require_audit and a.require_mix_audit and a.require_data_inventory_ready):
        ap.error("post-training phase requires token, mix and inventory audits")
    if a.append_manifest is not None and a.exclusion_manifest is not None:
        ap.error("--append-manifest and --exclusion-manifest are mutually exclusive")
    if (a.append_manifest is not None or a.exclusion_manifest is not None) and (a.fresh or a.init_adapter is not None or a.merge_only):
        ap.error("--append-manifest is only valid when resuming a copied checkpoint; it cannot start, reset, or merge-only")
    if (a.require_data_inventory_ready is None) != (a.inventory_policy is None):
        ap.error("--require-data-inventory-ready and --inventory-policy must be supplied together")
    if a.require_mix_audit is not None and not a.require_audit:
        ap.error("--require-mix-audit requires --require-audit")
    if (a.require_mix_audit is None) != (a.require_data_inventory_ready is None):
        ap.error("joint curriculum gates require both mix and inventory reports")
    if a.require_mix_audit is not None and not 0 < a.required_reducer_share < 1:
        ap.error("--required-reducer-share must be between zero and one")
    if (a.repair_mean_nll is None) != (a.repair_token_nll is None):
        ap.error("online repair requires both NLL thresholds")
    if a.repair_mean_nll is not None and any(not math.isfinite(v) or v <= 0 for v in (a.repair_mean_nll, a.repair_token_nll)):
        ap.error("repair NLL thresholds must be finite and positive")
    if a.repair_full_gold_every < 0:
        ap.error("--repair-full-gold-every must be nonnegative")
    audit_manifest = None
    if a.require_audit:
        try:
            audit_manifest = validate_training_audit(a.data, a.max_len, a.model, a.model_revision)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
    joint_gate_identity = None
    if a.require_mix_audit is not None:
        try:
            mix_identity = validate_training_mix_audit(
                a.require_mix_audit, a.data, a.required_reducer_share, audit_manifest)
            inventory_identity = validate_training_inventory_audit(
                a.require_data_inventory_ready, a.inventory_policy)
            joint_gate_identity = {**mix_identity, **inventory_identity}
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
    stop = install_stop_handlers()
    if a.steps is not None and a.steps < 1:
        ap.error("--steps must be positive")
    if a.epochs is not None and a.epochs <= 0:
        ap.error("--epochs must be positive")
    if a.periodic_heldout_every < 0 or a.periodic_heldout_count < 1:
        ap.error("periodic held-out interval must be nonnegative and subset size positive")
    if a.periodic_heldout_at_start and not a.periodic_heldout_every:
        ap.error("--periodic-heldout-at-start requires --periodic-heldout-every")
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
    if a.merge_only and a.no_merge:
        ap.error("--no-merge cannot be combined with --merge-only")
    if a.init_adapter is not None and not a.init_adapter.is_dir():
        ap.error("--init-adapter must be an adapter directory")
    targets = [x.strip() for x in a.target_modules.split(",") if x.strip()] if a.target_modules else None
    if a.unsloth_moe and (a.unsloth or a.unsloth_lfm_experts):
        ap.error("--unsloth-moe is for the Transformers loader; Unsloth's loader brings its MoE support itself")
    if a.exclude_modules and (a.unsloth or a.unsloth_lfm_experts):
        ap.error("--exclude-modules applies to the PEFT path, not Unsloth")
    if a.target_modules and not targets:
        ap.error("--target-modules must name at least one module")
    default_targets = (["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"]
                       if a.unsloth_lfm_experts else ["q_proj", "k_proj", "v_proj", "o_proj"])
    ckpt = a.out / "checkpoint"
    recover_checkpoint_directory(a.out)
    state_file = ckpt / "state.json"
    if a.fresh and ckpt.exists():
        shutil.rmtree(ckpt)
    resume = state_file.exists()
    if a.append_manifest is not None and not resume:
        ap.error("--append-manifest requires an existing copied parent checkpoint")
    state = json.loads(state_file.read_text()) if resume else {"step": 0, "cursor": 0, "skipped": 0, "log": []}
    if a.profile and a.benchmark_steps < 2:
        ap.error("--profile needs --benchmark-steps 2 or more: the earlier steps warm up kernels and allocator")
    if a.benchmark_steps and (resume or a.merge_only):
        ap.error("benchmarks require a separate output directory without a checkpoint")
    if a.merge_only and not resume:
        raise SystemExit(f"no checkpoint in {ckpt}")

    phase_receipt = None
    phase_first_transition = False
    if a.phase_manifest:
        if not resume: ap.error("post-training phase requires copied checkpoint")
        from scripts.posttraining_phase import verify_phase
        try:
            phase_receipt, phase_first_transition = verify_phase(a.phase_manifest, a.data, ckpt,
                seed=a.seed, holdout=a.holdout, lr=a.lr, accum=a.accum,
                online_repair=({"version":1,"mean_nll":a.repair_mean_nll,"token_nll":a.repair_token_nll,
                    "full_gold_every":a.repair_full_gold_every,"boundary":"complete-assistant-action"}
                    if a.repair_mean_nll is not None else None))
        except (OSError, ValueError, KeyError) as exc:
            ap.error(f"invalid post-training phase: {exc}")
    append_receipt = None
    append_manifest_sha256 = None
    append_first_transition = False
    exclusion_receipt = None
    exclusion_manifest_sha256 = None
    if not a.merge_only:
        pairs = index_pairs(a.data)
        held, train, split = split_programs(pairs, a.holdout, a.seed)
        train = order_training_pairs(train, a.data_order)
        if a.append_manifest is not None:
            try:
                append_receipt = json.loads(a.append_manifest.read_text(encoding="utf-8"))
                append_manifest_sha256 = file_digest(a.append_manifest)
                if append_receipt.get("schema") != "natlang.training_append_intake/2":
                    raise ValueError("unsupported append manifest schema")
                if append_receipt.get("combined", {}).get("path") != str(a.data.resolve()):
                    raise ValueError("append manifest names a different combined corpus")
                if append_receipt.get("combined", {}).get("sha256") != file_digest(a.data):
                    raise ValueError("combined corpus hash differs from append manifest")
                base_path = Path(append_receipt["base"]["path"])
                candidate_path = Path(append_receipt["candidate"]["path"])
                if file_digest(base_path) != append_receipt["base"]["sha256"]:
                    raise ValueError("base corpus hash differs from append manifest")
                if file_digest(candidate_path) != append_receipt["candidate"]["sha256"]:
                    raise ValueError("candidate corpus hash differs from append manifest")
                base_pairs = index_pairs(base_path)
                candidate_pairs = index_pairs(candidate_path)
                base_held, base_train, base_split = split_programs(
                    base_pairs, a.holdout, a.seed)
                base_train = order_training_pairs(base_train, a.data_order)
                if digest(base_split) != append_receipt["base"]["split_sha256"]:
                    raise ValueError("base split no longer matches append manifest")
                combined_held, combined_train, combined_split = split_programs(
                    base_pairs + candidate_pairs, a.holdout, a.seed)
                base_held_ids = {row["id"] for row in base_held}
                combined_held_ids = {row["id"] for row in combined_held}
                if base_held_ids != combined_held_ids:
                    raise ValueError("append changes the protected held-out row set")
                if any(row.get("split") != "train" for row in candidate_pairs):
                    raise ValueError("append candidate rows must be explicitly train")
                train = append_train_order(
                    base_train, candidate_pairs, pairs,
                    base_order_sha256=append_receipt["append_order"]["base_train_order_ids_sha256"],
                    candidate_order_sha256=append_receipt["append_order"]["candidate_order_ids_sha256"],
                )
                split = {**combined_split, "append_order": {
                    "version": 1,
                    "base_split_sha256": append_receipt["base"]["split_sha256"],
                    "base_train_order_ids_sha256": append_receipt["append_order"]["base_train_order_ids_sha256"],
                    "candidate_order_ids_sha256": append_receipt["append_order"]["candidate_order_ids_sha256"],
                }}
                if len(train) != len(base_train) + len(candidate_pairs):
                    raise ValueError("append train order row count is inconsistent")
                if len(base_train) != append_receipt["append_order"]["base_train_rows"]:
                    raise ValueError("base train row count differs from append manifest")
                if len(candidate_pairs) != append_receipt["append_order"]["candidate_rows"]:
                    raise ValueError("candidate row count differs from append manifest")
            except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
                ap.error(f"invalid --append-manifest: {exc}")
        if a.exclusion_manifest is not None:
            try:
                exclusion_receipt = json.loads(a.exclusion_manifest.read_text(encoding="utf-8"))
                exclusion_manifest_sha256 = file_digest(a.exclusion_manifest)
                if exclusion_receipt.get("schema") != "natlang.training_source_exclusion/1" or exclusion_receipt.get("status") != "approved":
                    raise ValueError("exclusion manifest is not an approved version-1 transition")
                if not resume or append_receipt is not None:
                    raise ValueError("exclusion requires a copied parent checkpoint and no append transition")
                if exclusion_receipt.get("data", {}).get("sha256") != file_digest(a.data):
                    raise ValueError("exclusion manifest names different corpus bytes")
                excluded_ids = exclusion_receipt.get("excluded_row_ids")
                source_review = exclusion_receipt.get("source_review", {})
                source_review_path = Path(source_review.get("path", ""))
                if (not source_review_path.is_absolute() or file_digest(source_review_path) != source_review.get("sha256")):
                    raise ValueError("source-review policy bytes differ from the approved exclusion manifest")
                source_review_text = source_review_path.read_text(encoding="utf-8")
                if (source_review.get("status") != "pending" or
                        not pending_source_review_entry(source_review_text, str(source_review.get("source_id")))):
                    raise ValueError("source-review policy no longer contains the exact pending hold")
                extra_holds = source_review.get("additional_pending_sources", [])
                descriptor_pin = source_review.get("additional_exclusions_descriptor")
                if extra_holds:
                    if not isinstance(descriptor_pin, dict):
                        raise ValueError("additional source holds lack a pinned exclusion descriptor")
                    descriptor_path = Path(descriptor_pin.get("path", ""))
                    if (not descriptor_path.is_absolute() or
                            file_digest(descriptor_path) != descriptor_pin.get("sha256")):
                        raise ValueError("additional source-exclusion descriptor changed")
                    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
                    rows_by_source = {str(item.get("source_id")): item
                                      for item in descriptor.get("rows", [])}
                    if not isinstance(excluded_ids, list):
                        raise ValueError("exclusion row identities are missing")
                    for held_source_review in extra_holds:
                        held_id = str(held_source_review.get("source_id"))
                        held_row = str(held_source_review.get("row_id"))
                        described = rows_by_source.get(held_id, {})
                        if (not pending_source_review_entry(source_review_text, held_id) or
                                held_row not in excluded_ids or
                                described.get("row_id") != held_row or
                                described.get("source_id") != held_id or
                                described.get("source_snapshot_sha256") != held_source_review.get("source_snapshot_sha256") or
                                described.get("source_program_id") != held_source_review.get("source_program_id") or
                                described.get("corpus_row_sha256") != held_source_review.get("corpus_row_sha256")):
                            raise ValueError(f"additional source hold is not exactly excluded: {held_id}")
                if exclusion_receipt.get("split", {}).get("identity_sha256") != digest(split):
                    raise ValueError("exclusion manifest names a different original split")
                parent = exclusion_receipt.get("checkpoint", {})
                parent_identity = parent.get("parent_corpus_identity")
                if not isinstance(parent_identity, dict) or parent_identity != state.get("exclusion_transition", {}).get("parent_corpus_identity"):
                    raise ValueError("copied checkpoint transition does not match the approved parent corpus")
                if parent.get("step") != state.get("exclusion_transition", {}).get("parent_step"):
                    raise ValueError("copied checkpoint transition step differs from the approved parent")
                if state.get("exclusion_transition", {}).get("manifest_sha256") != exclusion_manifest_sha256:
                    raise ValueError("copied checkpoint is not bound to this exact exclusion manifest")
                if digest(excluded_ids) != exclusion_receipt.get("excluded_row_ids_sha256"):
                    raise ValueError("exclusion row-ID list hash differs from approved manifest")
                order = filter_training_order(train, excluded_ids, int(parent["cursor"]))
                if order.previous_order_sha256 != exclusion_receipt.get("training_order", {}).get("previous_ids_sha256"):
                    raise ValueError("original training permutation differs from approved exclusion manifest")
                active_split = exclusion_split_manifest(split, order.rows, excluded_ids)
                if digest(active_split) != exclusion_receipt.get("split", {}).get("active_identity_sha256"):
                    raise ValueError("exclusion manifest active split does not match filtered training rows")
                mix_gate = exclusion_receipt.get("input_gates", {}).get("post_exclusion_mix", {})
                actual_mix = post_exclusion_mix_summary(
                    a.data, train, excluded_ids, float(mix_gate.get("target_reducer_share", -1)))
                if actual_mix != mix_gate or not actual_mix.get("target_met"):
                    raise ValueError("post-exclusion reducer mix differs from the approved source-exclusion gate")
                if order.filtered_order_sha256 != exclusion_receipt.get("training_order", {}).get("filtered_ids_sha256"):
                    raise ValueError("filtered training permutation differs from approved exclusion manifest")
                train = order.rows
                split = active_split
                target_examples = int(exclusion_receipt["target"]["examples"])
                expected_target = exclusion_target(int(parent_identity["target_examples"]),
                                                   int(parent["trained_examples"]),
                                                   len(order.future_removed_ids))
                if target_examples != expected_target or state.get("corpus", {}).get("target_examples") != target_examples:
                    raise ValueError("exclusion target does not preserve consumed history and remaining clean rows")
                preserved_steps = int(parent_identity["steps"])
                if exclusion_receipt["target"].get("steps_horizon_preserved") != preserved_steps:
                    raise ValueError("exclusion manifest changes the original scheduler horizon")
                a.steps = preserved_steps
                actual_checkpoint_files = {str(path.relative_to(ckpt)): file_digest(path)
                                           for path in ckpt.rglob("*") if path.is_file()
                                           and path.relative_to(ckpt).as_posix() != "state.json"}
                validate_exclusion_checkpoint_resume(
                    exclusion_receipt, state, actual_checkpoint_files,
                    accumulation=a.accum, manifest_sha256=exclusion_manifest_sha256)
                marker = state.get("exclusion_transition", {})
                if (marker.get("cursor") != order.cursor or marker.get("target_examples") != target_examples or
                        marker.get("parent_state_sha256") != parent.get("state_sha256") or
                        marker.get("parent_cursor") != parent.get("cursor") or
                        marker.get("consumed_removed_ids") != list(order.consumed_removed_ids) or
                        marker.get("future_removed_ids") != list(order.future_removed_ids)):
                    raise ValueError("copied checkpoint exclusion marker is inconsistent")
            except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
                ap.error(f"invalid --exclusion-manifest: {exc}")
        else:
            target_examples = (max(1, math.ceil(len(train) * a.epochs))
                               if a.epochs is not None else (a.steps or 300) * a.accum)
        if phase_receipt is not None:
            target_examples = phase_receipt["target_examples"]
        if append_receipt is not None:
            expected_target = extended_target_examples(
                int(append_receipt["checkpoint"]["corpus_identity"]["target_examples"]),
                len(index_pairs(Path(append_receipt["candidate"]["path"]))))
            if target_examples != expected_target:
                ap.error(f"append transition must extend the existing example target to {expected_target}; got {target_examples}")
        if phase_receipt is not None:
            a.steps = phase_receipt["end_step"]
        elif exclusion_receipt is None:
            a.steps = math.ceil(target_examples / a.accum)
        elif a.steps != int(exclusion_receipt["target"]["steps_horizon_preserved"]):
            ap.error("source exclusion must retain the original scheduler horizon")
        identity = {"data_sha256": file_digest(a.data), "split_sha256": digest(split),
                    "max_len": a.max_len, "model": a.model,
                    "model_revision": a.model_revision, "accum": a.accum,
                    "microbatch": a.microbatch, "batch_tokens": a.batch_tokens,
                    "seed": a.seed, "data_order": a.data_order,
                    "target_examples": target_examples, "steps": a.steps, "lr": a.lr,
                    "full": a.full, "rank": a.rank,
                    "target_modules": targets,
                    "load_in_4bit": a.load_in_4bit,
                    "unsloth": a.unsloth,
                    "unsloth_lfm_experts": a.unsloth_lfm_experts,
                    "unsloth_compile": a.unsloth_compile,
                    "gradient_checkpointing": a.gradient_checkpointing,
                    "checkpoint_above_tokens": a.checkpoint_above_tokens,
                    "require_audit": a.require_audit}
        if a.repair_mean_nll is not None:
            identity["online_repair"] = {"version":1, "mean_nll":a.repair_mean_nll,
                "token_nll":a.repair_token_nll, "full_gold_every":a.repair_full_gold_every, "boundary":"complete-assistant-action"}
        if exclusion_receipt is not None:
            identity["exclusion_manifest_sha256"] = exclusion_manifest_sha256
        if append_receipt is not None:
            identity["append_manifest_sha256"] = append_manifest_sha256
        if phase_receipt is not None:
            identity["phase_manifest_sha256"] = file_digest(a.phase_manifest)
        if joint_gate_identity is not None:
            identity["joint_gate_identity"] = joint_gate_identity
        if a.exclude_modules:
            identity["exclude_modules"] = a.exclude_modules
        if a.no_expert_lora:
            identity["no_expert_lora"] = True
        if a.expert_rank:
            identity["expert_rank"] = a.expert_rank
        if a.unsloth_moe:
            identity["unsloth_moe"] = True
        if a.optimizer != "adamw":
            identity["optimizer"] = a.optimizer
        if a.optimizer == "muon":
            if not 0 <= a.muon_momentum < 1 or a.muon_ns_steps < 1:
                ap.error("Muon requires momentum in [0,1) and positive Newton-Schulz steps")
            identity["muon"] = {"momentum": a.muon_momentum, "ns_steps": a.muon_ns_steps,
                                "adjust_lr_fn": "match_rms_adamw", "partition_version": 1}
        if a.retain_every_n_layers:
            identity["retain_every_n_layers"] = a.retain_every_n_layers
        if a.device != "cuda":
            identity["device"] = a.device
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
        if exclusion_receipt is not None and state.get("corpus") != identity:
            raise SystemExit("copied exclusion checkpoint corpus identity differs from the approved transition")
        validate_heldout_rows(held, split)
        if resume:
            old_identity = state.get("corpus")
            transition = state.get("append_transition")
            if phase_first_transition:
                permitted = {"data_sha256", "split_sha256", "target_examples", "steps", "lr",
                             "exclusion_manifest_sha256", "append_manifest_sha256", "phase_manifest_sha256", "joint_gate_identity", "online_repair"}
                if {k:v for k,v in old_identity.items() if k not in permitted} != {k:v for k,v in identity.items() if k not in permitted}:
                    raise SystemExit("Phase cannot change model/tokenizer/optimizer/training architecture")
                for k in ("mix_policy_sha256", "target_reducer_share", "policy_sha256"):
                    if old_identity.get("joint_gate_identity",{}).get(k) != identity.get("joint_gate_identity",{}).get(k):
                        raise SystemExit("Phase cannot change curriculum admission policy")
                state.setdefault("phase_history", []).append({"parent_state":phase_receipt["parent"]["state"],
                    "manifest_sha256":file_digest(a.phase_manifest)})
                state["cursor"] = 0
                state.pop("heldout_after", None)
                state.pop("heldout_after_step", None)
            elif old_identity == identity:
                if append_receipt is not None and (not transition or transition.get("manifest_sha256") != append_manifest_sha256):
                    raise SystemExit("Append checkpoint is not bound to this exact append manifest.")
                if append_receipt is None and transition:
                    raise SystemExit("This checkpoint requires its pinned --append-manifest to reconstruct data order.")
            else:
                parent = append_receipt.get("checkpoint", {}) if append_receipt else {}
                first_transition = (append_receipt is not None
                                    and old_identity == parent.get("corpus_identity")
                                    and file_digest(state_file) == parent.get("sha256"))
                if not first_transition:
                    raise SystemExit("Checkpoint corpus/split/settings differ without an exact authorized append transition.")
                permitted_changes = {"data_sha256", "split_sha256", "target_examples", "steps",
                                     "append_manifest_sha256", "joint_gate_identity"}
                comparable = {key: value for key, value in identity.items() if key not in permitted_changes}
                parent_comparable = {key: value for key, value in old_identity.items()
                                     if key not in permitted_changes}
                if comparable != parent_comparable:
                    raise SystemExit("Append cannot change model, optimizer, tokenizer, or training settings.")
                old_joint = old_identity.get("joint_gate_identity", {})
                new_joint = identity.get("joint_gate_identity", {})
                if old_joint.get("mix_policy_sha256") != new_joint.get("mix_policy_sha256"):
                    raise SystemExit("Append cannot change the reducer-mix audit policy.")
                if old_joint.get("target_reducer_share") != new_joint.get("target_reducer_share"):
                    raise SystemExit("Append cannot change the reducer-mix target.")
                old_inventory = old_joint.get("policy_sha256")
                new_inventory = new_joint.get("policy_sha256")
                if old_inventory != new_inventory:
                    raise SystemExit("Append cannot change the data-inventory policy.")
                expected_files = parent.get("files")
                if not isinstance(expected_files, dict) or not expected_files:
                    raise SystemExit("Append manifest lacks checkpoint artifact pins.")
                actual_files = {str(path.relative_to(ckpt)) for path in ckpt.rglob("*") if path.is_file()}
                if actual_files != set(expected_files):
                    raise SystemExit("Copied checkpoint file set differs from append manifest.")
                if any(file_digest(ckpt / name) != expected for name, expected in expected_files.items()):
                    raise SystemExit("Copied checkpoint artifacts differ from append manifest.")
                append_first_transition = True
                state.pop("heldout_after", None)
                state.pop("heldout_after_step", None)
        state["corpus"] = identity
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "split.json").write_text(json.dumps(split, indent=2) + "\n")
        print(f"program split: {len(train)} training turns, {len(held)} held-out turns "
              f"from {len(split['held_programs'])} programs; target {target_examples} examples "
              f"({(target_examples - state.get('trained_examples',state['step']*a.accum)) / len(train):.3g} remaining epochs over this corpus, "
              f"{a.steps} global optimizer steps)", flush=True)
    if stop["now"]:
        print("stop requested before model loading; no optimizer step was started", flush=True)
        return
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
    elif a.unsloth_moe:  # Its MoE patches to Transformers and PEFT apply on import, before either is used.
        if a.moe_backend:
            os.environ["UNSLOTH_MOE_BACKEND"] = a.moe_backend
        import unsloth  # noqa: F401
        if a.moe_backend:
            # Unsloth picks the backend once, while it is still importing, when its own Triton kernels cannot be
            # imported yet; both answers are cached. Ask again now that they can.
            from unsloth_zoo.temporary_patches import moe_utils
            moe_utils._GROUPED_GEMM_AVAILABLE = None
            moe_utils.select_moe_backend.cache_clear()
            if moe_utils.select_moe_backend() != a.moe_backend:
                raise RuntimeError(f"MoE backend {a.moe_backend} is not available here")
        if a.gradient_checkpointing and a.offload_checkpoints:  # as Unsloth's loader does: layer inputs wait in CPU memory
            from unsloth_zoo.gradient_checkpointing import patch_unsloth_smart_gradient_checkpointing
            patch_unsloth_smart_gradient_checkpointing(dtype=torch.bfloat16)
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not 0 < a.cuda_memory_fraction <= 1:
        raise ValueError("CUDA memory fraction must lie in (0, 1]")
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.cuda_memory_fraction)
    if device == "cpu" and (a.load_in_4bit or use_unsloth):
        raise ValueError("4-bit and Unsloth training require CUDA")

    base_src = str(ckpt / "weights") if (resume and a.full) else a.model
    if use_unsloth:
        from huggingface_hub import snapshot_download
        model_path = Path(a.model)
        base_src = (str(model_path.resolve()) if model_path.is_dir() else
                    snapshot_download(a.model, revision=a.model_revision))
        model, tok = FastLanguageModel.from_pretrained(
            model_name=base_src, max_seq_length=a.max_len, load_in_4bit=True, device_map=0)
        if audit_manifest is not None:
            validate_training_audit_tokenizer(audit_manifest, tok, a.model, a.model_revision)
        if a.unsloth_lfm_experts:
            restore_lfm_expert_quantization(model, base_src)
    else:
        tok = AutoTokenizer.from_pretrained(a.model, revision=a.model_revision, trust_remote_code=a.trust_remote_code)
        if audit_manifest is not None:
            validate_training_audit_tokenizer(audit_manifest, tok, a.model, a.model_revision)
        load_options = {"dtype": torch.bfloat16 if device == "cuda" else torch.float32,
                        "trust_remote_code": a.trust_remote_code}
        if a.model_revision:
            load_options["revision"] = a.model_revision
        if a.load_in_4bit:
            load_options.update({"device_map": {"": 0}, "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)})
        model = AutoModelForCausalLM.from_pretrained(base_src, **load_options)
        if not a.load_in_4bit:
            model = model.to(device)
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if not a.full:
        # PEFT's preparation upcasts every unquantized weight to fp32, and with them the activations; Unsloth
        # trains in bf16, and fp32 would double the embeddings and each dequantized expert stack.
        if a.load_in_4bit and not a.unsloth_lfm_experts and not a.unsloth_moe and not a.no_kbit_upcast:
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
                # With any layer family filtered out (a text model leaves vision out), Unsloth keeps only the
                # listed modules it finds under an attention or MLP block, silently dropping others such as LFM's
                # conv projections. The list is already exact; take it as given.
                model = FastLanguageModel.get_peft_model(
                    model, r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                    target_modules=linear, use_gradient_checkpointing="unsloth", random_state=a.seed,
                    **({"rank_pattern": {EXPERTS: a.expert_rank}, "alpha_pattern": {EXPERTS: 2 * a.expert_rank}}
                       if a.expert_rank else {}),
                    finetune_vision_layers=True, finetune_language_layers=True,
                    finetune_attention_modules=True, finetune_mlp_modules=True)
            else:
                # Stacked expert weights are parameters, not layers; PEFT adapts them through target_parameters.
                layers, stacked = split_targets(model, linear)
                if a.no_expert_lora:
                    stacked = []
                model = get_peft_model(model, LoraConfig(
                    r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                    target_modules=layers, target_parameters=stacked or None, exclude_modules=a.exclude_modules, task_type="CAUSAL_LM",
                    **({"rank_pattern": {EXPERTS: a.expert_rank}, "alpha_pattern": {EXPERTS: 2 * a.expert_rank}}
                       if a.expert_rank else {})))
            require_adapter_coverage(model, linear, a.exclude_modules)
        model.enable_input_require_grads()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)", flush=True)
        if device == "cuda":
            print(f"model memory: {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

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

    repair_metadata = {}
    repair_outbox = None
    if a.repair_mean_nll is not None:
        from scripts.online_repair import RepairOutbox, gated_loss
        repair_outbox = RepairOutbox(a.out / "repair-outbox.sqlite", {
            "data_sha256": state["corpus"]["data_sha256"],
            "policy": identity["online_repair"], "tokenizer":digest(tok.backend_tokenizer.to_str())})

    def encode(p, phase="train"):        # already rendered by the chat template: no special tokens added
        offset = p["offset"]
        cached = cache.get(offset) if cache else None
        if cached is None:
            data_stream.seek(offset)
            p = json.loads(data_stream.readline())
            family = p.get("family", "unknown")
            if "segments" in p:   # a merged conversation (scripts/merge_sft_chains.py)
                cached = ["segments", [[tok(text, add_special_tokens=False)["input_ids"], trained]
                                       for text, trained in p["segments"]], family]
            else:
                masked = p.get("completion_masked", 0)   # reasoning no model wrote is context, not a target
                cached = [tok(p["prompt"] + p["completion"][:masked], add_special_tokens=False)["input_ids"],
                          tok(p["completion"][masked:], add_special_tokens=False)["input_ids"], family]
            if cache:
                cache.put(offset, cached)
        counts = state.setdefault("overlength_encounters", {}).setdefault(phase, {})
        if cached[0] == "segments":
            # Each completion is trained with everything before it, so a later turn is only ever longer: keep the
            # turns up to the last that fits, as those turns alone would have been kept.
            _, segments, family = cached
            sequence, trained, turns, action_ends = [], [], 0, []
            for tokens, is_completion in segments:
                if is_completion and len(sequence) + len(tokens) > a.max_len:
                    break
                sequence += tokens
                trained += [is_completion] * len(tokens)
                turns += is_completion
                if is_completion: action_ends.append(len(sequence))
            dropped = sum(1 for _, is_completion in segments if is_completion) - turns
            if dropped:
                counts[family] = counts.get(family, 0) + dropped
            if not turns:
                return None
            while not trained[-1]:
                sequence.pop(); trained.pop()
            first = trained.index(True)
            result = (sequence[:first], sequence[first:], trained[first:])
            if repair_outbox is not None and phase == "train":
                repair_metadata[id(result)] = (offset, action_ends)
            return result
        x, y, family = cached
        if not x or not y:
            raise ValueError("Training pairs require a nonempty prompt and completion")
        if len(x) + len(y) > a.max_len:
            if a.require_audit:
                raise ValueError(f"audited training example at offset {offset} exceeds --max-len {a.max_len}")
            counts[family] = counts.get(family, 0) + 1
            return None
        result = (x, y)
        if repair_outbox is not None and phase == "train":
            repair_metadata[id(result)] = (offset, [len(x)+len(y)])
        return result

    @torch.no_grad()
    def heldout_loss():
        eval_rng = capture_rng_state()
        model.eval()
        tot = n = 0
        try:
            for p in held[:100]:
                if stop["now"]:
                    break
                e = encode(p, phase="heldout")
                if e:
                    tot += batch_completion_loss(model, collate_completions([e], device=device)).item(); n += 1
        finally:
            model.train()
            restore_rng_state(eval_rng)
        if stop["now"]:
            return None
        if held and not n:
            raise ValueError("Every held-out example exceeds --max-len")
        return tot / n if n else None

    if a.periodic_heldout_every and not held:
        ap.error("periodic held-out evaluation needs at least one held-out row")
    periodic_rows = (select_fixed_subset(held, a.periodic_heldout_count, a.seed)
                     if a.periodic_heldout_every else [])
    if periodic_rows:
        # Hydrate labels only for this fixed observational subset. The indexed
        # rows, split, ordering, rendered bytes, and training examples stay intact.
        periodic_rows = hydrate_support_metadata(a.data, periodic_rows)
    periodic_ids_sha256 = canonical_sha256([str(row["id"]) for row in periodic_rows])
    periodic_metrics_path = a.out / "heldout-periodic.jsonl"
    if a.periodic_heldout_every:
        print(f"periodic held-out loss: every {a.periodic_heldout_every} steps, "
              f"{len(periodic_rows)} deterministic rows ({periodic_ids_sha256[:16]}), "
              f"metrics {periodic_metrics_path}", flush=True)
    base_model_path = Path(a.model)
    base_model_files_sha256 = directory_digest(base_model_path) if base_model_path.is_dir() else None
    periodic_identity = {
        "data_sha256": state["corpus"]["data_sha256"],
        "split_sha256": state["corpus"]["split_sha256"],
        "model_revision": a.model_revision,
        "model_files_sha256": base_model_files_sha256,
        "model_identity_sha256": canonical_sha256({"model": a.model,
                                                    "revision": a.model_revision,
                                                    "files_sha256": base_model_files_sha256}),
    }
    try:
        periodic_completed = (read_periodic_metrics(
            periodic_metrics_path, data_sha256=periodic_identity["data_sha256"],
            split_sha256=periodic_identity["split_sha256"],
            model_identity_sha256=periodic_identity["model_identity_sha256"])
            if a.periodic_heldout_every else {})
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        ap.error(f"invalid periodic held-out metrics: {exc}")

    def run_periodic_eval_if_due(*, force=False):
        step = int(state["step"])
        if (not a.periodic_heldout_every or step < 1 or
                (not force and step % a.periodic_heldout_every)):
            return False
        key = (step, periodic_ids_sha256)
        weights_sha256 = trainable_weights_sha256(model)
        if periodic_metric_already_complete(
                periodic_completed, step=step, subset_ids_sha256=periodic_ids_sha256,
                trainable_weights_sha256=weights_sha256):
            return False
        if stop["now"]:
            return False

        def encode_periodic(row):
            example = encode(row, phase="periodic-heldout")
            if example is None:
                return None
            return collate_completions([example], pad_id=tok.pad_token_id or 0, device=device)

        def periodic_token_loss(m, encoded):
            supervised_tokens = int((encoded["labels"][:, 1:] != -100).sum().item())
            if not supervised_tokens:
                return 0.0, 0
            mean_loss = batch_completion_loss(m, encoded)
            return float(mean_loss.item()) * supervised_tokens, supervised_tokens

        metrics = evaluate_fixed_subset(
            model=model, rows=periodic_rows, encode=encode_periodic,
            token_loss_sum=periodic_token_loss, capture_rng=capture_rng_state,
            restore_rng=restore_rng_state, should_stop=lambda: stop["now"])
        if metrics is None:
            return False
        item = {
            "schema": "natlang.periodic_heldout_loss/1", "status": "completed",
            "model": a.model, **periodic_identity,
            "step": step, "subset_count": len(periodic_rows),
            "subset_selection": "sha256-rank-v1", "subset_seed": a.seed,
            "evaluation_trigger": "resume-start" if force else "step-interval",
            "subset_ids": [str(row["id"]) for row in periodic_rows],
            "subset_ids_sha256": periodic_ids_sha256,
            "trainable_weights_sha256": weights_sha256,
            **metrics,
            "metrics": {
                "heldout_loss_token_weighted": metrics["loss"],
                "heldout_supervised_tokens": metrics["supervised_tokens"],
                "heldout_evaluated_examples": metrics["evaluated_examples"],
                "heldout_skipped_examples": metrics["skipped_examples"],
                "heldout_class_count": len(metrics["class_support"]),
                "heldout_source_count": len(metrics["source_support"]),
                "heldout_source_group_count": len(metrics["source_group_support"]),
            },
        }
        append_periodic_metric(periodic_metrics_path, item)
        periodic_completed[key] = item
        print(f"periodic held-out loss at step {step}: {metrics['loss']:.6f} "
              f"({metrics['supervised_tokens']} masked assistant tokens; "
              f"{metrics['evaluated_examples']} rows)", flush=True)
        return True

    trained = [p for p in model.parameters() if p.requires_grad]
    if a.optimizer == "muon":
        from scripts.training_optimizers import make_muon_optimizer
        opt = make_muon_optimizer(model, lr=a.lr, momentum=a.muon_momentum,
                                  ns_steps=a.muon_ns_steps)
        print("Muon parameter partition: " + json.dumps({
            kind: sum(item["optimizer"] == kind for item in opt.schema)
            for kind in ("muon", "adamw")}), flush=True)
    elif a.optimizer == "paged-adamw-8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(trained, lr=a.lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(trained, lr=a.lr, weight_decay=0.0)
    scheduler_extension = state.get("scheduler_extension")
    def lr_multiplier(s):
        if scheduler_extension is not None:
            return cosine_extension_multiplier(
                s, start_step=scheduler_extension["start_step"],
                end_step=scheduler_extension["end_step"])
        return min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_multiplier)
    if resume:
        opt.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location=device))
        sched.load_state_dict(torch.load(ckpt / "scheduler.pt"))
        restore_rng_state(torch.load(ckpt / "rng.pt", map_location="cpu", weights_only=False))
        if phase_first_transition:
            if int(sched.last_epoch) != int(state["step"]):
                raise SystemExit("Parent scheduler step differs from optimizer step")
            start_lrs = [a.lr for _ in opt.param_groups]
            scheduler_extension = {"kind":"posttraining_cosine/1", "start_step":int(state["step"]),
                "end_step":a.steps, "start_lrs":start_lrs}
            sched.base_lrs = start_lrs
            for group,lr in zip(opt.param_groups,start_lrs):
                group["lr"] = lr; group["initial_lr"] = lr
            sched._last_lr = start_lrs
            state["scheduler_extension"] = scheduler_extension
            state["posttraining_phase"] = {"manifest_sha256":file_digest(a.phase_manifest),
                "at_step":int(state["step"]), "parent_cursor":phase_receipt["parent"]["state"]["cursor"],
                "optimizer_state_preserved":True, "rng_state_preserved":True,
                "scheduler_extension":scheduler_extension}
            # Prior append/exclusion states remain in the immutable parent lineage;
            # the new phase order is governed by its own manifest.
            state.pop("append_transition",None)
        elif append_first_transition:
            if state.get("append_transition"):
                raise SystemExit("checkpoint already contains an append transition")
            if int(sched.last_epoch) != int(state["step"]):
                raise SystemExit("checkpoint scheduler step differs from optimizer step")
            current_lrs = [float(group["lr"]) for group in opt.param_groups]
            if len(current_lrs) != len(sched.base_lrs) or any(not math.isfinite(lr) or lr < 0 for lr in current_lrs):
                raise SystemExit("checkpoint optimizer learning rates are invalid for append")
            start_step = int(state["step"])
            if start_step >= a.steps:
                raise SystemExit("append scheduler has no remaining steps")
            scheduler_extension = {"kind": "cosine_continuation/1", "start_step": start_step,
                                   "end_step": a.steps, "start_lrs": current_lrs,
                                   "previous_end_step": int(append_receipt["checkpoint"]["corpus_identity"]["steps"])}
            sched.base_lrs = current_lrs
            for group, lr in zip(opt.param_groups, current_lrs):
                group["initial_lr"] = lr
            state["scheduler_extension"] = scheduler_extension
            state["append_transition"] = {
                "manifest_sha256": append_manifest_sha256,
                "parent_state_sha256": append_receipt["checkpoint"]["sha256"],
                "at_step": start_step,
                "cursor": int(state["cursor"]),
                "scheduler_extension": scheduler_extension,
                "optimizer_state_preserved": True,
                "rng_state_preserved": True,
            }
        elif scheduler_extension is not None:
            start_lrs = scheduler_extension.get("start_lrs", [])
            if len(start_lrs) != len(opt.param_groups):
                raise SystemExit("saved scheduler extension does not match optimizer groups")
            sched.base_lrs = [float(lr) for lr in start_lrs]
        print(f"resumed from step {state['step']} (pair {state['cursor']})", flush=True)
    else:
        torch.manual_seed(a.seed)
        state["heldout_before"] = (None if a.benchmark_steps or a.skip_heldout_loss or stop["now"]
                                    else heldout_loss())
        print(f"{len(train)} training pairs; held-out loss before: {state['heldout_before']}", flush=True)

    def save_checkpoint():
        write_checkpoint_directory(
            a.out,
            lambda weights: model.save_pretrained(weights, safe_serialization=True),
            opt.state_dict(), sched.state_dict(), capture_rng_state(),
            {**state, "args": {k: str(v) for k, v in vars(a).items()}},
        )
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

    a.out.mkdir(parents=True, exist_ok=True)
    model.train()
    if device == "cuda":
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
            "packages": packages, "device": device,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else 0,
            "steps": metrics}, indent=2) + "\n")
        tmp.replace(metrics_file)

    def save_metrics_at_boundary():
        try:
            save_metrics()
        except BaseException:
            if not a.benchmark_steps:
                save_checkpoint()
            raise

    target_steps = a.benchmark_steps or a.steps
    target_examples = a.benchmark_steps * a.accum if a.benchmark_steps else state["corpus"]["target_examples"]
    state.setdefault("trained_examples", state["step"] * a.accum)
    if resume and not a.benchmark_steps:
        # A crash after the durable optimizer checkpoint but before its due evaluation
        # leaves a due step; finish it before advancing training.
        if a.periodic_heldout_every and (
                a.periodic_heldout_at_start or state["step"] % a.periodic_heldout_every == 0):
            run_periodic_eval_if_due(force=a.periodic_heldout_at_start)
    while state["step"] < target_steps and state["trained_examples"] < target_examples and not stop["now"]:
        boundary = (state["cursor"], state["skipped"], state["trained_examples"])
        rng_boundary = capture_rng_state()
        optimizer_started = False
        profiler = None
        if a.profile and state["step"] == target_steps - 1:
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                          torch.profiler.ProfilerActivity.CUDA], with_stack=True)
            profiler.__enter__()
        step_start = time.perf_counter()
        try:
            examples = []
            repair_metadata.clear()
            repair_observations = []
            for _ in range(min(a.accum, target_examples - state["trained_examples"])):
                e = None
                start_cursor = state["cursor"]
                while e is None:
                    if state["cursor"] - start_cursor >= len(train):
                        raise ValueError("Every training example exceeds --max-len")
                    e = encode(train[state["cursor"] % len(train)]); state["cursor"] += 1
                    state["skipped"] += e is None
                examples.append(e)
            full_gold_flags = {id(e): bool(a.repair_full_gold_every and
                (state["trained_examples"] + i + 1) % a.repair_full_gold_every == 0)
                for i,e in enumerate(examples)}
            ready = time.perf_counter()
            running = torch.zeros((), device=device)
            batches = 0
            padded_tokens = 0
            for batch in microbatches(examples, a.microbatch, a.batch_tokens):
                encoded = collate_completions(batch, pad_id=tok.pad_token_id or 0, device=device)
                want_checkpointing = a.gradient_checkpointing and encoded["input_ids"].numel() > a.checkpoint_above_tokens
                if want_checkpointing != model.is_gradient_checkpointing:
                    set_layer_checkpointing(model, want_checkpointing, a.retain_every_n_layers)
                if repair_outbox is None:
                    loss = batch_completion_loss(model, encoded)
                else:
                    # Loss column zero predicts input position min_prompt_len.
                    label_origin = min(len(e[0]) for e in batch)
                    ends = [[end-label_origin for end in repair_metadata[id(e)][1]] for e in batch]
                    loss, receipts = gated_loss(model, encoded, ends,
                        mean_nll=a.repair_mean_nll, token_nll=a.repair_token_nll,
                        full_gold=[full_gold_flags[id(e)] for e in batch])
                    for e,receipt in zip(batch,receipts):
                        receipt["label_origin"] = label_origin
                        receipt["prompt_tokens"] = len(e[0])
                    repair_observations.extend((repair_metadata[id(e)][0], receipt)
                                               for e,receipt in zip(batch,receipts))
                loss = loss * (len(batch) / len(examples))
                require_finite_loss(loss, state["step"] + 1)
                loss.backward()
                running += loss.detach()
                padded_tokens += encoded["input_ids"].numel()
                batches += 1
            clip_finite_grad_norm_(trained, 1.0)
            optimizer_started = True
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            state["step"] += 1
            state["trained_examples"] += len(examples)
            if repair_outbox is not None:
                repair_outbox.record_step(state["step"], repair_observations)
        except BaseException:
            # A failed preparation/backward has not committed an optimizer step.
            # Restore the cursor and counters to the prior boundary and persist that state.
            opt.zero_grad(set_to_none=True)
            state["cursor"], state["skipped"], state["trained_examples"] = boundary
            if not a.benchmark_steps and not optimizer_started:
                restore_rng_state(rng_boundary)
                save_checkpoint()
            raise
        running = running.item()  # one synchronization per optimizer step, not per sequence
        if profiler is not None:
            torch.cuda.synchronize()
            profiler.__exit__(None, None, None)
            print(profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=45, max_name_column_width=70),
                  flush=True)
            # The same, by the Python code that issued the operators.
            print(profiler.key_averages(group_by_stack_n=6).table(sort_by="self_cuda_time_total", row_limit=25,
                                                                  max_name_column_width=40, max_src_column_width=120),
                  flush=True)
        if cache:
            cache.db.commit()
        metrics.append({"step": state["step"], "seconds": time.perf_counter() - step_start,
                        "prepare_seconds": ready - step_start, "examples": len(examples),
                        "tokens": sum(len(e[0]) + len(e[1]) for e in examples),
                        "completion_tokens": sum(len(e[1]) if len(e) == 2 else sum(e[2]) for e in examples),
                        "padded_tokens": padded_tokens, "microbatches": batches, "loss": running,
                        **({"repair_flagged":sum(r["cutoff"] is not None for _,r in repair_observations),
                            "repair_unchanged":sum(r["cutoff"] is None for _,r in repair_observations),
                            "repair_retained_tokens":sum(r["retained_tokens"] for _,r in repair_observations),
                            "repair_total_tokens":sum(r["supervised_tokens"] for _,r in repair_observations)}
                           if repair_outbox is not None else {})})
        if state["step"] % 10 == 0 or state["step"] == target_steps:
            save_metrics_at_boundary()
            state["log"].append([state["step"], round(running, 4)])
            print(f"step {state['step']:4d}  loss {running:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"mem {(torch.cuda.max_memory_allocated() / 2**30 if device == 'cuda' else 0):.1f} GiB  {time.time() - t0:.0f}s "
                  f"overlength={state.get('overlength_encounters', {})}", flush=True)
        if not a.benchmark_steps and state["step"] % a.save_every == 0:
            save_checkpoint()
        if not a.benchmark_steps and a.periodic_heldout_every and state["step"] % a.periodic_heldout_every == 0:
            if state["step"] % a.save_every:
                save_checkpoint()
            run_periodic_eval_if_due()
    save_metrics_at_boundary()
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
    state["heldout_after"] = (None if a.skip_heldout_loss or stop["now"] else heldout_loss())
    state["heldout_after_step"] = state["step"] if state["heldout_after"] is not None else None
    if stop["now"]:
        save_checkpoint()
        print(f"stopped at step {state['step']} of {a.steps}; checkpoint written", flush=True)
        return
    print(f"held-out loss after: {state['heldout_after']}; trained examples {state['trained_examples']}, "
          f"source rows visited {state['cursor']}, too long {state['skipped']}", flush=True)
    save_checkpoint()
    if a.no_merge:
        print(f"saved checkpoint {ckpt}; merged export skipped", flush=True)
    elif a.load_in_4bit:
        print(f"saved adapter {ckpt / 'weights'}; QLoRA checkpoints are not merged into their quantized base", flush=True)
    else:
        export_merged()
    if cache:
        cache.db.commit()
        cache.db.close()
    data_stream.close()


if __name__ == "__main__":
    main()
