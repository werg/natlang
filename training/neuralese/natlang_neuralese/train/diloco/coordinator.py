"""Durable, authenticated, HTTP-reachable Decoupled DiLoCo coordinator.

Ported from megatoken (``megatoken.training.diloco_server``) without its
transducer bank. No GPU-to-GPU connectivity is required: learners send
outbound HTTP(S) only. Every committed fragment version carries the
deduplication sequence ledger, so replay after a coordinator crash cannot
apply a learner's update twice. Behind a CoreWeave sandbox share-token
gateway, this server speaks plain HTTP on a private port.

Root layout::

    manifest.json        immutable run description (written last at init)
    seed.safetensors     every trainable master at version 0 (learner bootstrap)
    fragments/NNNNN      authoritative FP32 theta + outer momentum + last_seq
    public/NNNNN         theta only, what learners download
    pending/NNNNN/WWW    queued pseudo-gradients awaiting a quorum
    progress.json        leases and per-learner high-water counters
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import torch
from safetensors import safe_open
from safetensors.torch import save

from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import write_bytes_durable, write_json_durable
from natlang_neuralese.train.diloco.protocol import (
    PROTOCOL,
    Fragment,
    ema_decay_at,
    load_tensors,
    merge_pseudogradients,
    outer_nesterov,
    pack_fragment,
    parameter_identity,
    plan_fragments,
    save_tensors,
)

TOKEN_ENV = "NATLANG_DILOCO_TOKEN"
TOKEN_SECRET = "diloco-token"


def credential(env: str, secret: str) -> str:
    """A credential from the environment, else from the secret file a CoreWeave launch delivered."""
    value = os.environ.get(env, "")
    directory = os.environ.get("NATLANG_CW_SECRETS")
    if not value and directory and (Path(directory) / secret).is_file():
        value = (Path(directory) / secret).read_text().strip()
    return value


class Conflict(ValueError):
    """A stale or duplicated submission, or a live learner holding the slot (HTTP 409).

    ``detail`` rides along in the body: ``stale`` for a base the coordinator
    cannot reconcile, ``lease`` when the learner's own lease ran out.
    """

    def __init__(self, message: str, **detail: object) -> None:
        super().__init__(message)
        self.detail = detail


def _metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as file:
        return file.metadata() or {}


def _manifest_fingerprint(manifest: dict) -> str:
    return canonical_json_sha256_hex({k: v for k, v in manifest.items() if k != "fingerprint"})


def read_manifest(root: str | Path) -> dict:
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    if _manifest_fingerprint(manifest) != manifest.get("fingerprint"):
        raise ValueError("Coordinator manifest fingerprint mismatch")
    return manifest


def initialize_coordinator(
    root: str | Path,
    *,
    run_id: str,
    masters: Mapping[str, torch.Tensor],
    identity: Mapping,
    workers: int,
    min_quorum: int,
    fragment_bytes: int = 64 << 20,
    outer_lr: float = 0.7,
    outer_momentum: float = 0.9,
    grace_seconds: float = 8.0,
    max_staleness: int = 2,
    lease_seconds: float = 600.0,
    token_budget: int = 0,
    seed_source: Mapping | None = None,
    ema: Mapping | None = None,
    outer_schedule: list | None = None,
) -> dict:
    """Create a new run from the seed masters; never overwrite an existing trajectory.

    ``identity`` is the trainer's objective identity (model, recipe, data
    split, objective). Learners must present exactly the same mapping.
    ``seed_source`` records where the seed weights came from (path and sha).
    ``ema``: ``{"decay": d, "warmup": N}`` keeps an exponential moving average of the *global versions*: each merged
    fragment version v+1 updates ``ema <- ema + (1 - decay_v) (theta_{v+1} - ema)`` in the same atomic commit as the
    version itself (``decay_v``: the warmup schedule at v earlier merges), starting from the seed. The coordinator
    holds its masters and outer momentum in FP32, so its EMA is FP32 too; ``native_diloco export --ema-out`` casts
    it to each parameter's dtype. Absent, manifests and fragments are exactly as before.
    ``outer_schedule``: ``[{"from_round": 0, "lr": 1.0, "momentum": 0.0}, {"from_round": 10, "lr": 0.7,
    "momentum": 0.9}]`` replaces the single ``outer_lr``/``outer_momentum`` (``outer_entry``): a fragment's merge number
    v (its version before the merge) uses the last entry with ``from_round <= v``, and the outer momentum buffer
    restarts at zero when the entry changes. It is part of the manifest and so of the run fingerprint.
    """
    root = Path(root)
    if outer_schedule is not None:
        outer_schedule = validate_outer_schedule(outer_schedule)
        outer_lr, outer_momentum = outer_schedule[0]["lr"], outer_schedule[0]["momentum"]
    if ema is not None:
        decay, warmup = ema.get("decay"), ema.get("warmup", 0)
        if (
            set(ema) - {"decay", "warmup"}
            or type(decay) not in (int, float)
            or not 0 < decay < 1
            or type(warmup) is not int
            or warmup < 0
            or warmup == 1
        ):
            raise ValueError("Invalid DiLoCo EMA configuration")
        ema = {"decay": float(decay), "warmup": warmup}
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", run_id)
        or not 1 <= min_quorum <= workers <= 256
        or not 0 <= grace_seconds <= 3600
        or not 0 <= max_staleness <= 32
        or not 0 < lease_seconds <= 86400
        or token_budget < 0
        or not (0 < outer_lr <= 4 and 0 <= outer_momentum < 1)
    ):
        raise ValueError("Invalid DiLoCo coordinator configuration")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a DiLoCo coordinator run at {root}")
    root.mkdir(parents=True, exist_ok=True)
    fragments = plan_fragments(masters, target_bytes=fragment_bytes)
    manifest = {
        "protocol": PROTOCOL,
        "run_id": run_id,
        "workers": workers,
        "min_quorum": min_quorum,
        "fragment_bytes": fragment_bytes,
        "outer_lr": outer_lr,
        "outer_momentum": outer_momentum,
        "grace_seconds": grace_seconds,
        "max_staleness": max_staleness,
        "lease_seconds": lease_seconds,
        "token_budget": token_budget,
        "identity": dict(identity),
        "identity_sha256": canonical_json_sha256_hex(dict(identity)),
        "parameter_identity": parameter_identity(masters),
        "parameter_shapes": {name: list(p.shape) for name, p in sorted(masters.items())},
        "parameter_dtypes": {name: str(p.dtype).removeprefix("torch.") for name, p in sorted(masters.items())},
        "fragments": [item.to_dict() for item in fragments],
        "seed_source": dict(seed_source or {}),
        **({"ema": ema} if ema is not None else {}),
        **({"outer_schedule": outer_schedule} if outer_schedule is not None else {}),
    }
    manifest["fingerprint"] = _manifest_fingerprint(manifest)
    try:
        save_tensors(root / "seed.safetensors", {name: p.detach().float().cpu() for name, p in masters.items()})
        for fragment in fragments:
            values = pack_fragment(masters, fragment)
            save_tensors(
                root / "fragments" / f"{fragment.index:05d}.safetensors",
                {"theta": values, "momentum": torch.zeros_like(values), **({"ema": values.clone()} if ema else {})},
                metadata={"version": "0", "last_seq": "{}"},
            )
            save_tensors(
                root / "public" / f"{fragment.index:05d}.safetensors", {"theta": values}, metadata={"version": "0"}
            )
        write_json_durable(root / "progress.json", {"workers": {}, "leases": {}, "merges": 0})
        # The manifest is the last write: a run is discoverable only when complete.
        write_json_durable(root / "manifest.json", manifest)
    except BaseException:
        (root / "manifest.json").unlink(missing_ok=True)
        raise
    return manifest


def validate_outer_schedule(schedule) -> list[dict]:
    """A nonempty list of {from_round, lr, momentum}, starting at round 0 with strictly increasing rounds."""
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("outer schedule is a nonempty list of {from_round, lr, momentum}")
    out = []
    for entry in schedule:
        if not isinstance(entry, dict) or set(entry) != {"from_round", "lr", "momentum"}:
            raise ValueError("outer schedule entries are exactly {from_round, lr, momentum}")
        start, lr, momentum = entry["from_round"], entry["lr"], entry["momentum"]
        if (type(start) is not int or start < 0 or type(lr) not in (int, float) or type(momentum) not in (int, float)
                or not (0 < lr <= 4 and 0 <= momentum < 1)):
            raise ValueError("invalid outer schedule entry: " + json.dumps(entry, sort_keys=True))
        out.append({"from_round": start, "lr": float(lr), "momentum": float(momentum)})
    if out[0]["from_round"] != 0 or any(b["from_round"] <= a["from_round"] for a, b in zip(out, out[1:])):
        raise ValueError("outer schedule starts at round 0 with strictly increasing from_round")
    return out


def outer_entry(manifest: Mapping, merge_round: int) -> tuple[int, dict]:
    """(index, entry) of the outer optimizer settings for a fragment's merge number ``merge_round`` (0-based)."""
    schedule = manifest.get("outer_schedule") or [
        {"from_round": 0, "lr": manifest["outer_lr"], "momentum": manifest["outer_momentum"]}]
    index = max(i for i, entry in enumerate(schedule) if entry["from_round"] <= merge_round)
    return index, schedule[index]


class Coordinator:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest = read_manifest(self.root)
        self.fragments = tuple(Fragment.from_dict(x) for x in self.manifest["fragments"])
        self.progress = json.loads((self.root / "progress.json").read_text())
        # ``lock`` serializes merges and file work, which can take long;
        # ``progress_lock`` guards only leases and counters, so heartbeats never
        # wait behind a merge (the lease would lapse meanwhile).
        self.lock = threading.RLock()
        self.progress_lock = threading.RLock()
        self.published_versions: list[int] = []
        self.quorum_since: dict[int, float] = {}
        self._sync_public()
        self._refresh_published()
        # A queued quorum surviving a restart gets an immediate opportunity to merge.
        self.tick()

    def _path(self, index: int) -> Path:
        if index < 0 or index >= len(self.fragments):
            raise ValueError("Unknown fragment index")
        return self.root / "fragments" / f"{index:05d}.safetensors"

    def _published(self, index: int) -> Path:
        return self.root / "public" / f"{index:05d}.safetensors"

    def _pending_dir(self, index: int) -> Path:
        return self.root / "pending" / f"{index:05d}"

    def _version(self, index: int) -> int:
        return int(_metadata(self._path(index))["version"])

    def _sync_public(self) -> None:
        """Reconcile public copies with the authoritative fragments after a crash."""
        with self.lock:
            for fragment in self.fragments:
                version = self._version(fragment.index)
                public = self._published(fragment.index)
                if not public.exists() or int(_metadata(public)["version"]) != version:
                    theta = load_tensors(self._path(fragment.index))["theta"]
                    save_tensors(public, {"theta": theta}, metadata={"version": str(version)})

    def _write_progress(self) -> None:
        write_json_durable(self.root / "progress.json", self.progress)

    def _refresh_published(self) -> None:
        self.published_versions = [self._version(f.index) for f in self.fragments]

    def _lease_valid(self, worker: int, session: str) -> None:
        with self.progress_lock:
            lease = self.progress["leases"].get(str(worker))
        if lease is None or lease["session"] != session or lease["expires"] < time.time():
            raise Conflict("Learner lease expired or belongs to a different session", lease=True)

    def lease(self, worker: int, session: str) -> dict:
        if not 0 <= worker < self.manifest["workers"] or not re.fullmatch(r"[a-f0-9-]{36}", session):
            raise ValueError("Invalid learner ID or session UUID")
        with self.progress_lock:
            previous = self.progress["leases"].get(str(worker))
            if previous and previous["expires"] > time.time() and previous["session"] != session:
                raise Conflict("Learner ID already leased by a live session")
            self.progress["leases"][str(worker)] = {
                "session": session,
                "expires": time.time() + self.manifest["lease_seconds"],
            }
            self.progress.setdefault("registered", {}).setdefault(str(worker), time.time())
            self._write_progress()
        with self.lock:
            return {**self.status(), "sequences": self._sequences(worker)}

    def _sequences(self, worker: int) -> list[int]:
        """The newest sequence held from ``worker`` per fragment (merged or queued).

        A learner restored from an older checkpoint numbers its next uploads
        after these, so none is mistaken for one already merged.
        """
        result = []
        for fragment in self.fragments:
            merged = json.loads(_metadata(self._path(fragment.index)).get("last_seq", "{}"))
            queued = self._pending_dir(fragment.index) / f"{worker:03d}.safetensors"
            result.append(
                max(int(merged.get(str(worker), 0)), int(_metadata(queued)["sequence"]) if queued.exists() else 0)
            )
        return result

    def heartbeat(self, worker: int, session: str, steps: int, tokens: int, metrics: dict | None = None) -> dict:
        if isinstance(metrics, dict) and isinstance(metrics.get("start_wait"), dict):
            with self.progress_lock:
                self.progress.setdefault("start_waits", {})[str(worker)] = metrics["start_wait"]
        with self.progress_lock:
            self._lease_valid(worker, session)
            if type(steps) is not int or type(tokens) is not int or steps < 0 or tokens < 0:
                raise ValueError("Progress counters must be nonnegative integers")
            previous = self.progress["workers"].get(str(worker), {"steps": 0, "tokens": 0})
            # High-water marks: a restored learner replaying windows is not credited twice.
            entry = {"steps": max(steps, previous["steps"]), "tokens": max(tokens, previous["tokens"])}
            if isinstance(metrics, dict):
                entry["metrics"] = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
                entry["seen"] = time.time()
            self.progress["workers"][str(worker)] = entry
            self.progress["leases"][str(worker)]["expires"] = time.time() + self.manifest["lease_seconds"]
            self._write_progress()
            return self._progress_status()

    def identity_transition(self, worker: int, session: str, transition: dict) -> dict:
        """Journal a locally verified source correction against this immutable run."""
        self._lease_valid(worker,session)
        from ..native_diloco import source_identity_transition
        if not isinstance(transition,dict) or transition.get('original_identity')!=self.manifest['identity']:
            raise ValueError('transition does not bind the original coordinator identity')
        current=transition.get('current_identity')
        checked=source_identity_transition(self.manifest['identity'],current,transition)
        if checked.get('worker')!=worker or checked.get('fingerprint')!=self.manifest['fingerprint']:
            raise ValueError('source transition belongs to a different worker or parameter surface')
        with self.progress_lock:
            self._lease_valid(worker,session)
            history=self.progress.setdefault('identity_transitions',{}).setdefault(str(worker),[])
            if checked in history:return {'recorded':True,'duplicate':True}
            if history and history[-1]['source_hash']!=checked['previous_source_hash']:
                raise Conflict('source identity transition does not continue its recorded predecessor')
            history.append(checked);self._write_progress()
        return {'recorded':True,'duplicate':False}

    def completion(self, worker: int, session: str, *, drained: bool, sequences: list[int] | None = None, generation: str | None = None) -> dict:
        """Record no-more-local-updates and, separately, delivery of each final window."""
        self._lease_valid(worker, session)
        if not 0 <= worker < self.manifest['workers'] or type(drained) is not bool:
            raise ValueError('invalid learner completion identity')
        if generation is not None and not re.fullmatch('[0-9a-f]{64}', generation):
            raise ValueError('invalid source generation')
        with self.lock:
            with self.progress_lock:
                done=self.progress.setdefault('completion',{})
                prior=done.get(str(worker))
                if prior and prior.get('generation') not in (None, generation):
                    raise Conflict('completion source generation changed without reopening')
                if drained:
                    if not prior or not prior.get('declared'):
                        raise ValueError('learner must declare completion before draining')
                    if (not isinstance(sequences,list) or len(sequences)!=len(self.fragments)
                            or any(type(x) is not int or x<0 for x in sequences)):
                        raise ValueError('invalid final learner fragment sequences')
                    held=self._sequences(worker)
                    if sequences != held:
                        raise Conflict('final learner updates have not been accepted', undelivered=True)
                done[str(worker)]={'declared':True,'drained':drained or bool(prior and prior.get('drained')),
                                   'sequences':sequences if drained else (prior or {}).get('sequences'),
                                   'generation':generation if generation is not None else (prior or {}).get('generation'), 'at':time.time()}
                self._write_progress()
                all_declared=len(done)==self.manifest['workers'] and all(v.get('declared') for v in done.values()) and len({v.get('generation') for v in done.values()})==1
            if all_declared:
                for fragment in self.fragments:self._try_merge(fragment.index,final=True)
                self._refresh_published()
            return self.status()

    def reopen(self, worker: int, session: str, *, previous: str, generation: str, sequences: list[int]) -> dict:
        """Open one previously drained worker for an explicitly bound new source generation."""
        self._lease_valid(worker, session)
        if not all(isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) for x in (previous,generation)) or previous==generation:
            raise ValueError('reopen needs distinct exact source generation hashes')
        with self.lock:
            with self.progress_lock:
                done=self.progress.setdefault('completion',{}); prior=done.get(str(worker))
                reopened=self.progress.setdefault('reopened',{})
                event={'previous':previous,'generation':generation,'sequences':sequences}
                if prior is None and reopened.get(str(worker))==event:
                    if sequences!=self._sequences(worker):
                        raise Conflict('reopened worker has newer coordinator updates; restore exact topology')
                    return self._progress_status()
                if not prior or not prior.get('drained'):
                    raise Conflict('only a fully drained worker can reopen')
                if prior.get('generation') not in (None, previous):
                    raise Conflict('prior source generation differs from completion custody')
                if (not isinstance(sequences,list) or sequences!=prior.get('sequences') or
                        sequences!=self._sequences(worker)):
                    raise Conflict('prior final sequences differ from coordinator custody')
                history=self.progress.setdefault('completion_history',{}).setdefault(str(worker),[])
                history.append({'completion':prior,'reopen':event,'at':time.time()})
                reopened[str(worker)]=event;del done[str(worker)];self._write_progress()
            return self.status()

    def _all_completion_declared(self) -> bool:
        with self.progress_lock:
            done=self.progress.get('completion') or {}
            return len(done)==self.manifest['workers'] and all(v.get('declared') for v in done.values()) and len({v.get('generation') for v in done.values()})==1

    def _progress_status(self) -> dict:
        with self.progress_lock:
            global_tokens = sum(x["tokens"] for x in self.progress["workers"].values())
            return {
                "run_id": self.manifest["run_id"],
                "global_tokens": global_tokens,
                "token_budget": self.manifest["token_budget"],
                "finished": bool(self.manifest["token_budget"] and global_tokens >= self.manifest["token_budget"]),
                "merges": self.progress["merges"],
                "versions": list(self.published_versions),
                "live_workers": sum(x["expires"] >= time.time() for x in self.progress["leases"].values()),
                "worker_progress": {k: dict(v) for k, v in self.progress["workers"].items()},
                "registered": dict(self.progress.get("registered", {})),
                "start_waits": dict(self.progress.get("start_waits", {})),
                "min_quorum": self.manifest["min_quorum"],
                "workers": self.manifest["workers"],
                "completion": dict(self.progress.get("completion") or {}),
                "recent_rounds": list(self.progress.get("rounds", []))[-8:],
                **({"outer_schedule": self.manifest["outer_schedule"],
                    "outer_next": [outer_entry(self.manifest, v)[1] for v in sorted(set(self.published_versions))]}
                   if "outer_schedule" in self.manifest else {}),
            }

    def status(self) -> dict:
        with self.lock:
            self._refresh_published()
            return {
                **self._progress_status(),
                "pending": {
                    str(f.index): len(list(self._pending_dir(f.index).glob("*.safetensors"))) for f in self.fragments
                },
            }

    def submit(
        self,
        index: int,
        worker: int,
        session: str,
        sequence: int,
        base_version: int,
        steps: int,
        tokens: int,
        incoming: Path,
    ) -> dict:
        with self.lock:
            self._lease_valid(worker, session)
            self._path(index)
            if any(type(v) is not int for v in (sequence, base_version, steps, tokens)) or (
                sequence < 1 or base_version < 0 or steps <= 0 or tokens <= 0
            ):
                raise ValueError("Invalid update metadata")
            fragment = self.fragments[index]
            version = self._version(index)
            last_seq = json.loads(_metadata(self._path(index)).get("last_seq", "{}"))
            if sequence <= last_seq.get(str(worker), 0):
                return {"status": "already_merged", "version": version}
            with self.progress_lock:
                completed = (self.progress.get("completion") or {}).get(str(worker), {})
            if completed.get("drained"):
                raise Conflict("Learner already declared its final updates drained")
            if base_version > version or version - base_version > self.manifest["max_staleness"]:
                raise Conflict("Stale fragment baseline; fetch current global weights", stale=True)
            try:
                tensors = load_tensors(incoming)
            except Exception as exc:
                raise ValueError("Malformed safetensors update") from exc
            if tensors.keys() != {"delta"}:
                raise ValueError("Expected exactly one safetensors delta")
            delta = tensors["delta"]
            if (
                delta.dtype not in (torch.float32, torch.bfloat16)
                or delta.shape != (fragment.length,)
                or not torch.isfinite(delta).all()
            ):
                raise ValueError("Delta dtype, shape, or finiteness invalid")
            target = self._pending_dir(index) / f"{worker:03d}.safetensors"
            if target.exists():
                if int(_metadata(target)["sequence"]) == sequence:
                    return {"status": "already_queued", "version": version}
                raise Conflict("One fragment update per learner may be in flight")
            save_tensors(
                target,
                {"delta": delta.float()},
                metadata={
                    "worker": str(worker),
                    "sequence": str(sequence),
                    "base_version": str(base_version),
                    "steps": str(steps),
                    "tokens": str(tokens),
                },
            )
            self._try_merge(index,final=self._all_completion_declared())
            return {"status": "queued", "version": self._version(index)}

    def _try_merge(self, index: int, *, final: bool = False) -> None:
        """Merge a queued quorum; ``final`` merges whatever is queued, however few."""
        path = self._path(index)
        meta = _metadata(path)
        version = int(meta["version"])
        last_seq = json.loads(meta.get("last_seq", "{}"))
        proposals = []
        for file in sorted(self._pending_dir(index).glob("*.safetensors")):
            item = _metadata(file)
            age = version - int(item["base_version"])
            if int(item["sequence"]) <= last_seq.get(item["worker"], 0) or not 0 <= age <= self.manifest["max_staleness"]:
                file.unlink()
                continue
            proposals.append((file, item, age))
        if len(proposals) < (1 if final else self.manifest["min_quorum"]):
            self.quorum_since.pop(index, None)
            return
        since = self.quorum_since.setdefault(index, time.monotonic())
        if (
            not final
            and len(proposals) < self.manifest["workers"]
            and time.monotonic() - since < self.manifest["grace_seconds"]
        ):
            return
        self.quorum_since.pop(index, None)
        data = load_tensors(path)
        delta = merge_pseudogradients(
            [load_tensors(file)["delta"] for file, _, _ in proposals],
            [int(item["tokens"]) for _, item, _ in proposals],
            [int(item["steps"]) for _, item, _ in proposals],
            [age for _, _, age in proposals],
        )
        entry_index, entry = outer_entry(self.manifest, version)
        velocity = data["momentum"]
        if version > 0 and outer_entry(self.manifest, version - 1)[0] != entry_index:
            velocity = torch.zeros_like(velocity)  # a new schedule entry starts its momentum buffer at zero
        theta, momentum = outer_nesterov(data["theta"], velocity, delta, lr=entry["lr"], momentum=entry["momentum"])
        for _, item, _ in proposals:
            last_seq[item["worker"]] = int(item["sequence"])
        committed = {"theta": theta, "momentum": momentum}
        settings = self.manifest.get("ema")
        if settings is not None:
            weight = 1.0 - ema_decay_at(settings["decay"], settings["warmup"], version)
            committed["ema"] = torch.lerp(data["ema"], theta, weight)
        # The authoritative fragment with its last_seq is the commit record;
        # proposals replayed after a crash are discarded by the ledger above.
        windows = {item["worker"]: {"steps": int(item["steps"]), "tokens": int(item["tokens"]), "staleness": age}
                   for _, item, age in proposals}
        record = {"version": str(version + 1), "last_seq": json.dumps(last_seq), "windows": json.dumps(windows)}
        if "outer_schedule" in self.manifest:
            record["outer"] = json.dumps({"entry": entry_index, **entry}, sort_keys=True)
        save_tensors(path, committed, metadata=record)
        save_tensors(self._published(index), {"theta": theta}, metadata={"version": str(version + 1)})
        # Heartbeats report these versions; a learner whose upload did not complete the quorum learns of the merge
        # only from them. Publishing on the reconcile tick (1 s) left it training against its upload point meanwhile,
        # unable to send its next round, so later merges combined unequal windows and overshot.
        if len(self.published_versions) == len(self.fragments):
            self.published_versions[index] = version + 1
        for file, _, _ in proposals:
            file.unlink(missing_ok=True)
        with self.progress_lock:
            self.progress["merges"] += 1
            # One record per (fragment, version); fragment 0 stands for the round in the rolling view.
            if index == 0:
                steps = [w["steps"] for w in windows.values()]
                rounds = self.progress.setdefault("rounds", [])
                rounds.append({"version": version + 1, "at": time.time(), "final": final, "windows": windows,
                               "step_imbalance": max(steps) / max(1, min(steps))})
                del rounds[:-256]
            self._write_progress()

    def tick(self) -> None:
        """Merge every fragment with a quorum; one failing merge does not starve the others."""
        errors: list[Exception] = []
        with self.lock:
            for fragment in self.fragments:
                try:
                    self._try_merge(fragment.index,final=self._all_completion_declared())
                except Exception as error:
                    errors.append(error)
            self._refresh_published()
        if errors:
            raise errors[0]

    def close(self) -> None:
        """Merge whatever learners sent last, however few, so the final window reaches the export."""
        with self.lock:
            errors: list[Exception] = []
            for fragment in self.fragments:
                try:
                    self._try_merge(fragment.index, final=True)
                except Exception as error:
                    errors.append(error)
            self._refresh_published()
            if errors:
                raise errors[0]


def _assemble(manifest: dict, fragment_values, label: str) -> dict[str, torch.Tensor]:
    """Full FP32 tensors from per-fragment flat values (``fragment_values(fragment) -> tensor``)."""
    out = {name: torch.empty(shape, dtype=torch.float32) for name, shape in manifest["parameter_shapes"].items()}
    for spec in manifest["fragments"]:
        fragment = Fragment.from_dict(spec)
        values = fragment_values(fragment)
        if values.dtype != torch.float32 or values.numel() != fragment.length:
            raise ValueError(f"Corrupt coordinator {label} fragment")
        cursor = 0
        for part in fragment.slices:
            out[part.name].reshape(-1)[part.offset : part.offset + part.length].copy_(
                values[cursor : cursor + part.length]
            )
            cursor += part.length
    return out


def _read_global(root: str | Path, key: str) -> tuple[dict[str, torch.Tensor], dict]:
    root = Path(root)
    manifest = read_manifest(root)
    if key == "ema" and manifest.get("ema") is None:
        raise ValueError("this DiLoCo run keeps no EMA of its global versions (declare ema at init)")
    versions = []

    def values(fragment):
        path = root / "fragments" / f"{fragment.index:05d}.safetensors"
        versions.append(int(_metadata(path)["version"]))
        return load_tensors(path)[key]

    tensors = _assemble(manifest, values, "EMA" if key == "ema" else "")
    info = {"versions": versions, "fingerprint": manifest["fingerprint"], "run_id": manifest["run_id"]}
    return tensors, {**info, **({"ema": manifest["ema"]} if key == "ema" else {})}


def global_masters(root: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    """The coordinator's current global FP32 masters, assembled from its fragments.

    Each fragment file is replaced atomically, so every fragment is read at one committed version (the versions are
    returned); a stopped coordinator gives one coherent version.
    """
    return _read_global(root, "theta")


def global_ema(root: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    """The EMA of the coordinator's global versions, assembled like ``global_masters`` (FP32)."""
    return _read_global(root, "ema")


def remote_global(transport, key: str, workdir: str | Path) -> tuple[dict[str, torch.Tensor], dict, dict]:
    """``global_masters``/``global_ema`` of a *running* coordinator over its HTTP API (``transport``: the learners'
    ``worker.Transport``), so an evaluation sandbox pulls global versions without the coordinator's volume. Fetches
    ``/v1/manifest`` and each committed fragment (``/v1/fragments/i`` for theta, ``/v1/ema/i`` for the EMA); every
    fragment is one committed version (recorded). Returns (tensors, info, manifest)."""
    manifest = transport.json("GET", "/v1/manifest")
    if key == "ema" and manifest.get("ema") is None:
        raise ValueError("this DiLoCo run keeps no EMA of its global versions (declare ema at init)")
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    versions = []
    route = "/v1/ema" if key == "ema" else "/v1/fragments"

    def values(fragment):
        path, headers = transport.fetch_file(f"{route}/{fragment.index}", workdir)
        if path is None:
            raise ValueError(f"coordinator returned no {key} fragment {fragment.index}")
        try:
            versions.append(int(headers["X-Fragment-Version"]))
            return load_tensors(path)["ema" if key == "ema" else "theta"]
        finally:
            path.unlink(missing_ok=True)

    tensors = _assemble(manifest, values, "EMA" if key == "ema" else "")
    info = {"versions": versions, "fingerprint": manifest["fingerprint"], "run_id": manifest["run_id"]}
    return tensors, {**info, **({"ema": manifest["ema"]} if key == "ema" else {})}, manifest


FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
HEX64 = re.compile(r"[0-9a-f]{64}")


class FileStore:
    """Named files beside a coordinator run (ROOT/uploads): continuation checkpoints and learner full states move
    sandbox to sandbox through the coordinator instead of through the workstation uplink.

    An upload streams into ``uploads/.partial/<name>.<sha256>`` (bounded memory) and may continue a partial file at
    its current size (``X-Offset``), so an interrupted transfer resumes. At the declared total size the partial file
    is hashed; a mismatch deletes it, a match is fsynced and renamed to ``uploads/<name>`` with a ``<name>.sha256``
    sidecar. A committed name is immutable: the same content again is ``present``, other content is a conflict."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / "uploads"
        self.partial = self.root / ".partial"
        self.locks: dict[str, threading.Lock] = {}
        self.guard = threading.Lock()

    @staticmethod
    def check(name: str, sha256: str | None = None) -> None:
        if not FILE_NAME.fullmatch(name) or name.endswith(".sha256"):
            raise ValueError("file names are 1-128 of [A-Za-z0-9._-], starting alphanumeric, not ending .sha256")
        if sha256 is not None and not HEX64.fullmatch(sha256):
            raise ValueError("X-Sha256 must be 64 lower-case hex digits")

    def lock(self, name: str) -> threading.Lock:
        with self.guard:
            return self.locks.setdefault(name, threading.Lock())

    def committed(self, name: str) -> str | None:
        sidecar = self.root / f"{name}.sha256"
        return sidecar.read_text().strip() if sidecar.is_file() and (self.root / name).is_file() else None

    def state(self, name: str, sha256: str) -> dict:
        self.check(name, sha256)
        done = self.committed(name)
        if done is not None:
            return {"status": "present" if done == sha256 else "conflict", "sha256": done,
                    "bytes": (self.root / name).stat().st_size}
        part = self.partial / f"{name}.{sha256}"
        return {"status": "partial", "bytes": part.stat().st_size if part.exists() else 0}

    def receive(self, name: str, sha256: str, offset: int, total: int, length: int, read) -> dict:
        self.check(name, sha256)
        if not (0 <= offset and 0 <= length and offset + length <= total):
            raise ValueError("X-Offset + Content-Length exceeds X-Total-Bytes")
        lock = self.lock(name)
        if not lock.acquire(blocking=False):
            raise Conflict("another upload of this name is in progress")
        try:
            done = self.committed(name)
            if done is not None:
                if done != sha256:
                    raise Conflict("a different file is already committed under this name")
                return {"status": "present", "sha256": done, "bytes": (self.root / name).stat().st_size}
            self.partial.mkdir(parents=True, exist_ok=True)
            part = self.partial / f"{name}.{sha256}"
            have = part.stat().st_size if part.exists() else 0
            if offset != have:
                raise Conflict("upload offset does not match the partial file", offset=have)
            with part.open("ab") as handle:
                remaining = length
                while remaining:
                    block = read(min(4 << 20, remaining))
                    if not block:
                        break  # a dropped client leaves a resumable prefix
                    handle.write(block)
                    remaining -= len(block)
                handle.flush()
                os.fsync(handle.fileno())
            size = part.stat().st_size
            if remaining:
                raise ValueError(f"truncated upload: {size} of {total} bytes kept, resume at X-Offset {size}")
            if size < total:
                return {"status": "partial", "bytes": size}
            digest = hashlib.sha256()
            with part.open("rb") as handle:
                for block in iter(lambda: handle.read(8 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != sha256:
                part.unlink()
                raise ValueError("sha256 mismatch: the upload was discarded")
            os.replace(part, self.root / name)
            write_bytes_durable(self.root / f"{name}.sha256", (sha256 + "\n").encode())
            return {"status": "committed", "sha256": sha256, "bytes": size}
        finally:
            lock.release()

    def open(self, name: str):
        self.check(name)
        sha256 = self.committed(name)
        if sha256 is None:
            raise FileNotFoundError(f"no committed file {name}")
        return (self.root / name).open("rb"), sha256


def serve_coordinator(root: str | Path, host: str, port: int, token: str, *, stop: threading.Event | None = None) -> None:
    """Serve on a private host, behind the sandbox gateway or an authenticated TLS proxy."""
    if not token or len(token) < 24:
        raise ValueError(f"Set {TOKEN_ENV} to a random secret (>=24 chars)")
    coordinator = Coordinator(root)
    files = FileStore(coordinator.root)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass  # never log authorization headers, sessions or query values

        def _auth(self) -> bool:
            offered = self.headers.get("Authorization", "").encode("utf-8", "replace")
            if hmac.compare_digest(offered, f"Bearer {token}".encode()):
                return True
            self._json(401, {"error": "Unauthorized"})
            return False

        def _json(self, status: int, value: dict) -> None:
            body = json.dumps(value, sort_keys=True, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _empty(self) -> None:
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _stream(self, file, version: int | None = None, sha256: str | None = None) -> None:
            with file:
                size = (file.getbuffer().nbytes if isinstance(file, io.BytesIO)
                        else os.fstat(file.fileno()).st_size)
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(size))
                self.send_header("Cache-Control", "no-store")
                if version is not None:
                    self.send_header("X-Fragment-Version", str(version))
                if sha256 is not None:
                    self.send_header("X-Sha256", sha256)
                self.end_headers()
                for chunk in iter(lambda: file.read(4 << 20), b""):
                    self.wfile.write(chunk)

        def _read_json(self, maximum: int = 65536) -> dict:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= maximum:
                raise ValueError("Invalid JSON payload size")
            result = json.loads(self.rfile.read(size))
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
            return result

        def _header(self, name: str) -> str:
            value = self.headers.get(name)
            if value is None:
                raise ValueError(f"Missing {name} header")
            return value

        def _receive(self, maximum: int) -> Path:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= maximum:
                raise ValueError("Oversized or empty update")
            tempdir = coordinator.root / "incoming"
            tempdir.mkdir(exist_ok=True)
            fd, filename = tempfile.mkstemp(dir=tempdir, suffix=".safetensors")
            try:
                with os.fdopen(fd, "wb") as handle:
                    remaining = length
                    while remaining:
                        block = self.rfile.read(min(4 << 20, remaining))
                        if not block:
                            raise ValueError("Truncated update upload")
                        handle.write(block)
                        remaining -= len(block)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                Path(filename).unlink(missing_ok=True)
                raise
            return Path(filename)

        def do_GET(self) -> None:
            if not self._auth():
                return
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/v1/manifest":
                    self._json(200, coordinator.manifest)
                elif parsed.path == "/v1/status":
                    self._json(200, coordinator.status())
                elif parsed.path.startswith("/v1/files/"):
                    name = parsed.path[len("/v1/files/"):]
                    query = parse_qs(parsed.query)
                    if "sha256" in query:  # upload state: committed, conflict, or the partial size to resume at
                        self._json(200, files.state(name, query["sha256"][0]))
                    else:
                        try:
                            file, sha256 = files.open(name)
                        except FileNotFoundError as exc:
                            self._json(404, {"error": str(exc)})
                            return
                        self._stream(file, sha256=sha256)
                elif parsed.path == "/v1/seed":
                    self._stream((coordinator.root / "seed.safetensors").open("rb"))
                elif re.fullmatch(r"/v1/seed-fragments/[0-9]+", parsed.path):
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    coordinator._path(index)  # Validate fragment index.
                    fragment = coordinator.fragments[index]
                    with safe_open(coordinator.root / "seed.safetensors", framework="pt", device="cpu") as seed:
                        pieces = [seed.get_tensor(part.name).reshape(-1)[part.offset:part.offset + part.length]
                                  for part in fragment.slices]
                        payload = io.BytesIO(save({"theta": torch.cat(pieces).float()}))
                    self._stream(payload, 0)
                elif re.fullmatch(r"/v1/ema/[0-9]+", parsed.path):
                    # The EMA of the global versions for one fragment (evaluation pulls; learners never read it).
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    if coordinator.manifest.get("ema") is None:
                        self._json(404, {"error": "this run keeps no EMA of its global versions"})
                    else:
                        with coordinator.lock:
                            path = coordinator._path(index)
                            version = int(_metadata(path)["version"])
                            ema = load_tensors(path)["ema"]
                        payload = io.BytesIO(save({"ema": ema}, metadata={"version": str(version)}))
                        self._stream(payload, version)
                elif re.fullmatch(r"/v1/fragments/[0-9]+", parsed.path):
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    with coordinator.lock:
                        version = coordinator._version(index)
                        unchanged = parse_qs(parsed.query).get("after") == [str(version)]
                        # Open under the lock, but never hold it while a slow client streams.
                        file = None if unchanged else coordinator._published(index).open("rb")
                    if file is None:
                        self._empty()
                    else:
                        self._stream(file, version)
                else:
                    self._json(404, {"error": "Unknown endpoint"})
            except (TypeError, ValueError, OSError) as exc:
                self._json(400, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self) -> None:
            if not self._auth():
                return
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/v1/lease":
                    value = self._read_json()
                    result = coordinator.lease(int(value["worker"]), str(value["session"]))
                elif parsed.path == "/v1/identity-transition":
                    value=self._read_json()
                    result=coordinator.identity_transition(int(value['worker']),str(value['session']),value['transition'])
                elif parsed.path == "/v1/completion":
                    value=self._read_json()
                    result=coordinator.completion(int(value['worker']),str(value['session']),
                                                  drained=value['drained'],sequences=value.get('sequences'),generation=value.get('generation'))
                elif parsed.path == "/v1/reopen":
                    value=self._read_json()
                    result=coordinator.reopen(int(value['worker']),str(value['session']),previous=value['previous'],generation=value['generation'],sequences=value['sequences'])
                elif parsed.path == "/v1/progress":
                    value = self._read_json()
                    result = coordinator.heartbeat(
                        int(value["worker"]), str(value["session"]), value["steps"], value["tokens"], value.get("metrics")
                    )
                elif re.fullmatch(r"/v1/updates/[0-9]+", parsed.path):
                    index = int(parsed.path.rsplit("/", 1)[-1])
                    coordinator._path(index)
                    filename = self._receive(coordinator.fragments[index].length * 4 + 8192)
                    try:
                        result = coordinator.submit(
                            index,
                            int(self._header("X-Worker")),
                            self._header("X-Session"),
                            int(self._header("X-Sequence")),
                            int(self._header("X-Base-Version")),
                            int(self._header("X-Steps")),
                            int(self._header("X-Tokens")),
                            filename,
                        )
                    finally:
                        filename.unlink(missing_ok=True)
                else:
                    self._json(404, {"error": "Unknown endpoint"})
                    return
                self._json(200, result)
            except Conflict as exc:
                self._json(409, {"error": str(exc), **exc.detail})
            except (KeyError, TypeError, ValueError, OSError) as exc:
                self._json(400, {"error": str(exc)})

        def do_PUT(self) -> None:
            if not self._auth():
                return
            parsed = urlsplit(self.path)
            try:
                if not parsed.path.startswith("/v1/files/"):
                    self._json(404, {"error": "Unknown endpoint"})
                    return
                length = int(self.headers.get("Content-Length", "-1"))
                result = files.receive(parsed.path[len("/v1/files/"):], self._header("X-Sha256"),
                                       int(self.headers.get("X-Offset", "0")), int(self._header("X-Total-Bytes")),
                                       length, self.rfile.read)
                self._json(200, result)
            except Conflict as exc:
                self._json(409, {"error": str(exc), **exc.detail})
            except (KeyError, TypeError, ValueError, OSError) as exc:
                self.close_connection = True  # the unread rest of a refused body must not be parsed as a request
                self._json(400, {"error": str(exc)})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.request_queue_size = 32
    stop = stop or threading.Event()

    def reconcile() -> None:
        while not stop.wait(1):
            try:
                coordinator.tick()
            except Exception:
                traceback.print_exc()  # a failed merge must not end the loop serving later rounds
                sys.stderr.flush()

    def watch_stop() -> None:
        stop.wait()
        server.shutdown()

    threads = [threading.Thread(target=reconcile, daemon=True), threading.Thread(target=watch_stop, daemon=True)]
    for thread in threads:
        thread.start()
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        stop.set()
        server.server_close()
        threads[0].join(timeout=2)
        coordinator.close()
