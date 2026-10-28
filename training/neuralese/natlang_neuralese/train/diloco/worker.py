"""Nonblocking DiLoCo learner: transport, restart state and overlapped fragment sync.

Ported from megatoken (``megatoken.training.diloco_client``) without its
transducer bank. Network work runs on an I/O pool; live parameters change
only on the training thread between optimizer updates (``after_step``), and
threads never touch CUDA tensors. A wall-clock heartbeat keeps the learner's
lease fresh however long one update or a checkpoint takes. Pending uploads
are embedded in ``state_dict`` so a replaced sandbox can resend them.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields
from email.message import Message
from pathlib import Path
from typing import Any, Mapping

import torch

from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.train.diloco.coordinator import TOKEN_ENV, TOKEN_SECRET, credential
from natlang_neuralese.train.diloco.protocol import (
    PROTOCOL,
    Fragment,
    ParameterSurface,
    install_fragment,
    load_tensors,
    pack_fragment,
    parameter_identity,
    save_tensors,
)

GATEWAY_TOKEN_ENV = "NATLANG_DILOCO_GATEWAY_TOKEN"
GATEWAY_SECRET = "diloco-gateway"
WORKER_ID_ENV = "NATLANG_DILOCO_WORKER_ID"
HEARTBEAT_SECONDS = 10.0


@dataclass(frozen=True)
class DilocoSettings:
    """Learner settings (a recipe stage's ``diloco`` block); worker_id may come from the sandbox env."""

    run_id: str
    workers: int
    coordinator_url: str | None = None  # absent only where the coordinator is being created
    worker_id: int | None = None
    local_steps: int = 64
    io_workers: int = 4
    timeout_seconds: float = 120.0
    max_offline_steps: int = 512
    wire_dtype: str = "bf16"
    allow_private_http: bool = False
    heartbeat_seconds: float = HEARTBEAT_SECONDS
    # Start barrier: a fresh learner waits before update 0 until every learner has registered, or until a quorum has
    # and this many seconds passed (sandboxes boot minutes apart; an early learner would otherwise run alone).
    start_grace_seconds: float = 600.0

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("DiLoCo needs a run_id")
        if not 1 <= self.workers <= 256:
            raise ValueError("DiLoCo workers must be in [1, 256]")
        if min(self.local_steps, self.io_workers) < 1:
            raise ValueError("DiLoCo local_steps and io_workers must be positive")
        if self.timeout_seconds <= 0 or self.max_offline_steps < 0 or self.heartbeat_seconds <= 0 \
                or self.start_grace_seconds < 0:
            raise ValueError("Invalid DiLoCo timeout/offline tolerance")
        if self.wire_dtype not in {"bf16", "fp32"}:
            raise ValueError("DiLoCo wire_dtype must be bf16 or fp32")
        if self.worker_id is not None and not 0 <= self.worker_id < self.workers:
            raise ValueError("Invalid DiLoCo worker_id")

    @property
    def resolved_worker_id(self) -> int:
        raw = self.worker_id if self.worker_id is not None else int(os.environ.get(WORKER_ID_ENV, "-1"))
        if not 0 <= raw < self.workers:
            raise ValueError(f"Set {WORKER_ID_ENV} in [0, workers) on each learner")
        return raw

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> DilocoSettings:
        unknown = set(mapping) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown DiLoCo settings: {sorted(unknown)}")
        return cls(**mapping)

    def identity(self) -> dict:
        """The settings that change what a learner computes (excludes URLs and worker IDs)."""
        return {"run_id": self.run_id, "workers": self.workers, "local_steps": self.local_steps, "wire_dtype": self.wire_dtype}


class RemoteError(RuntimeError):
    def __init__(self, status: int, reason: str) -> None:
        super().__init__(f"DiLoCo coordinator HTTP {status}: {reason}")
        self.status = status
        try:
            detail = json.loads(reason)
        except ValueError:
            detail = None
        self.detail: dict = detail if isinstance(detail, dict) else {}


class _SameOriginRedirects(urllib.request.HTTPRedirectHandler):
    """urllib resends every header, the bearer token included, to a redirect target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = urllib.parse.urlsplit(req.full_url)
        new = urllib.parse.urlsplit(urllib.parse.urljoin(req.full_url, newurl))
        if (old.scheme, old.netloc) != (new.scheme, new.netloc):
            raise urllib.error.HTTPError(req.full_url, code, "Refusing a redirect to another origin", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _Window:
    """At most ``length`` bytes of an open file from its current position, read in blocks (a streamed body)."""

    def __init__(self, handle, length: int) -> None:
        self.handle, self.remaining = handle, length

    def read(self, size: int = -1) -> bytes:
        size = self.remaining if size is None or size < 0 else min(size, self.remaining)
        block = self.handle.read(size) if size else b""
        self.remaining -= len(block)
        return block


class Transport:
    def __init__(self, url: str, token: str, timeout: float, *, allow_http: bool = False, gateway: str = "") -> None:
        self.url = url.rstrip("/")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
            raise ValueError("Coordinator must be an absolute HTTP(S) URL")
        if parsed.scheme == "http" and not (allow_http or parsed.hostname in {"127.0.0.1", "localhost", "::1"}):
            raise ValueError("Use HTTPS for the coordinator outside localhost")
        if not token or len(token) < 24:
            raise ValueError(f"{TOKEN_ENV} is missing or too short")
        self.token = token
        self.timeout = timeout
        # A coordinator behind the CoreWeave sandbox gateway also needs its share token.
        self.gateway = {"X-Sandbox-Share-Token": gateway} if gateway else {}
        self.opener = urllib.request.build_opener(_SameOriginRedirects)

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        file: Path | None = None,
        headers: dict[str, str] | None = None,
        output: Path | None = None,
    ) -> tuple[int, Any, Message]:
        if json_body is not None and file is not None:
            raise ValueError("Cannot send JSON and a binary body together")
        body = (
            json.dumps(json_body, sort_keys=True, allow_nan=False).encode()
            if json_body is not None
            else file.read_bytes() if isinstance(file, Path) else file
        )
        request = urllib.request.Request(
            self.url + path,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                **self.gateway,
                **({"Content-Type": "application/json"} if json_body is not None else {}),
                **(headers or {}),
            },
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                # Header lookup is case-insensitive: HTTP/2 gateways lowercase names.
                if output is not None and response.status == 200:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open("wb") as handle:
                        for block in iter(lambda: response.read(4 << 20), b""):
                            handle.write(block)
                        handle.flush()
                        os.fsync(handle.fileno())
                    return response.status, output, response.headers
                raw = response.read()
                value = json.loads(raw) if raw and response.headers.get_content_type() == "application/json" else raw
                return response.status, value, response.headers
        except urllib.error.HTTPError as error:
            raise RemoteError(error.code, error.read(4096).decode(errors="replace")) from error

    def json(self, method: str, path: str, value: dict | None = None) -> dict:
        _, result, _ = self.request(method, path, json_body=value)
        if not isinstance(result, dict):
            raise ValueError(f"Expected JSON coordinator response from {path}")
        return result

    def upload_file(self, source: Path, name: str, *, chunk_bytes: int = 1 << 30, attempts: int = 8) -> dict:
        """Stream ``source`` to the coordinator file store as ``name`` (PUT /v1/files/NAME), resuming at the
        stored partial size after an interruption; the server verifies the sha256 before committing."""
        import hashlib
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
        sha256, total = digest.hexdigest(), source.stat().st_size
        quoted = urllib.parse.quote(name, safe="")
        for attempt in range(attempts):
            try:
                state = self.json("GET", f"/v1/files/{quoted}?sha256={sha256}")
                if state["status"] == "present":
                    return {**state, "name": name}
                if state["status"] == "conflict":
                    raise ValueError(f"{name}: a different file is committed under this name")
                offset = int(state["bytes"])
                while True:
                    length = min(chunk_bytes, total - offset)
                    with source.open("rb") as handle:
                        handle.seek(offset)
                        _, result, _ = self.request("PUT", f"/v1/files/{quoted}", file=_Window(handle, length),
                            headers={"Content-Type": "application/octet-stream", "Content-Length": str(length),
                                     "X-Sha256": sha256, "X-Offset": str(offset), "X-Total-Bytes": str(total)})
                    if result["status"] in ("committed", "present"):
                        return {**result, "name": name}
                    offset = int(result["bytes"])
            except (OSError, RemoteError) as error:
                if isinstance(error, RemoteError) and error.status not in (409, 500, 502, 503, 504):
                    raise
                if attempt == attempts - 1:
                    raise
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError("unreachable")

    def download_file(self, name: str, target: Path) -> dict:
        """GET /v1/files/NAME into ``target`` (atomic rename), verifying the stored sha256."""
        import hashlib
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".partial")
        status, _, headers = self.request("GET", f"/v1/files/{urllib.parse.quote(name, safe='')}", output=temporary)
        if status != 200:
            raise ValueError(f"Unexpected HTTP status {status}")
        digest = hashlib.sha256()
        with temporary.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != headers.get("X-Sha256"):
            temporary.unlink()
            raise ValueError(f"{name}: downloaded bytes do not match the coordinator's sha256")
        os.replace(temporary, target)
        return {"name": name, "path": str(target), "sha256": digest.hexdigest(), "bytes": target.stat().st_size}

    def fetch_file(self, path: str, directory: Path) -> tuple[Path | None, Message]:
        fd, name = tempfile.mkstemp(dir=directory, suffix=".download")
        os.close(fd)
        target = Path(name)
        try:
            status, _, headers = self.request("GET", path, output=target)
            if status == 204:
                target.unlink()
                return None, headers
            if status != 200:
                raise ValueError(f"Unexpected HTTP status {status}")
            return target, headers
        except BaseException:
            target.unlink(missing_ok=True)
            raise


def transport_from_env(settings: DilocoSettings) -> Transport:
    if not settings.coordinator_url:
        raise ValueError("A DiLoCo learner needs its coordinator_url")
    return Transport(
        settings.coordinator_url,
        credential(TOKEN_ENV, TOKEN_SECRET),
        settings.timeout_seconds,
        allow_http=settings.allow_private_http,
        gateway=credential(GATEWAY_TOKEN_ENV, GATEWAY_SECRET),
    )


class DilocoLearner:
    """One learner per GPU sandbox, without process groups or blocking collectives.

    ``identity`` is the trainer's objective identity; it must equal the one
    the coordinator was initialized with, as must the parameter shapes.
    """

    def __init__(
        self,
        settings: DilocoSettings,
        *,
        surface: ParameterSurface,
        identity: Mapping,
        output: Path,
        transport: Transport | None = None,
    ) -> None:
        self.settings = settings
        self.surface = surface
        self.worker = settings.resolved_worker_id
        self.shard = (self.worker, settings.workers)
        self.transport = transport or transport_from_env(settings)
        self.manifest = self.transport.json("GET", "/v1/manifest")
        mismatches = [
            label
            for label, ok in (
                ("protocol", self.manifest["protocol"] == PROTOCOL),
                ("run_id", self.manifest["run_id"] == settings.run_id),
                ("workers", self.manifest["workers"] == settings.workers),
                ("parameters", self.manifest["parameter_identity"] == parameter_identity(surface.masters)),
                ("identity", self.manifest["identity_sha256"] == canonical_json_sha256_hex(dict(identity))),
            )
            if not ok
        ]
        if mismatches:
            raise ValueError(f"DiLoCo learner differs from the coordinator run in: {', '.join(mismatches)}")
        self.fragments = tuple(Fragment.from_dict(x) for x in self.manifest["fragments"])
        self.outbox = Path(output) / "outbox"
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.io = ThreadPoolExecutor(max_workers=settings.io_workers, thread_name_prefix="diloco")
        self.remote_versions: list[int] | None = None
        self.session = str(uuid.uuid4())
        count = len(self.fragments)
        self.versions = [-1] * count
        self.baselines: list[torch.Tensor | None] = [None] * count
        self.steps_since = [0] * count
        self.tokens_since = [0] * count
        self.sequences = [0] * count
        self.pending: dict[int, dict] = {}
        self.awaiting_global: set[int] = set()
        self.downloads: dict[int, Future] = {}
        self.heartbeat_seconds = min(settings.heartbeat_seconds, self.manifest["lease_seconds"] / 4)
        self.heartbeat_thread: threading.Thread | None = None
        self.stopping = threading.Event()
        self.heartbeats = 0
        self.heartbeats_seen = 0
        self.heartbeat_error = ""
        self.report: dict = {}
        self.local_steps = 0
        self.local_tokens = 0
        self.global_tokens = 0
        self.merges = 0
        self.installs = 0
        self.finished = False
        self.last_error = ""
        self.offline_steps = 0

    # -- lease and heartbeat -------------------------------------------------

    def _lease(self) -> dict:
        status = self.transport.json("POST", "/v1/lease", {"worker": self.worker, "session": self.session})
        self._update_status(status)
        return status

    def _with_lease(self, call: Any) -> Any:
        """Run a call; if our own lease ran out meanwhile, lease again and retry once."""
        try:
            return call()
        except RemoteError as exc:
            if exc.status != 409 or not exc.detail.get("lease"):
                raise
            self._lease()
            return call()

    def _adopt_ledger(self, status: dict) -> None:
        """Number new uploads after everything the coordinator already holds from this learner."""
        self.sequences = [max(a, b) for a, b in zip(self.sequences, status["sequences"], strict=True)]

    def _start_heartbeat(self) -> None:
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name="diloco-heartbeat", daemon=True)
        self.heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self.stopping.wait(self.heartbeat_seconds):
            try:
                body = {
                    "worker": self.worker,
                    "session": self.session,
                    "steps": self.local_steps,
                    "tokens": self.local_tokens,
                    "metrics": {**self.report, **({"start_wait": self.start_wait} if getattr(self, "start_wait", None) else {})},
                }
                self._update_status(self._with_lease(lambda: self.transport.json("POST", "/v1/progress", body)))
                self.heartbeat_error = ""
                self.heartbeats += 1
            except Exception as exc:
                # Never end the thread: a dead heartbeat lets the lease lapse mid-training.
                self.heartbeat_error = f"{type(exc).__name__}: {exc}"

    def _update_status(self, value: dict) -> None:
        self.global_tokens = int(value["global_tokens"])
        self.merges = int(value["merges"])
        self.finished = bool(value["finished"])
        versions = value.get("versions")
        if isinstance(versions, list) and len(versions) == len(self.fragments):
            self.remote_versions = [int(v) for v in versions]

    # -- start and resume ----------------------------------------------------

    def start(self) -> None:
        """Lease, then install the current global fragments over the locally loaded parent.

        Every learner loads the same parent checkpoint (weights and inner
        optimizer); installing the coordinator's fragments makes the starting
        weights exactly the global ones, whatever version the run is at. A
        fragment still at version 0 is the coordinator's seed, which must equal
        the local parent exactly.
        """
        self._adopt_ledger(self._lease())
        self._start_heartbeat()
        for index in range(len(self.fragments)):
            update = self._fetch(index, -1)
            if update is None:
                raise ValueError(f"DiLoCo coordinator omitted fragment {index}")
            seed = update[1]
            if update[0] != 0:
                path, _ = self.transport.fetch_file(f"/v1/seed-fragments/{index}", self.outbox)
                if path is None:
                    raise ValueError(f"DiLoCo coordinator omitted seed fragment {index}")
                try:
                    seed = load_tensors(path)["theta"].float()
                finally:
                    path.unlink(missing_ok=True)
            if not torch.equal(seed, pack_fragment(self.surface.masters, self.fragments[index])):
                raise ValueError(f"DiLoCo seed fragment {index} differs from this learner's parent weights")
            self._install(index, update)

    def await_update_capacity(self, *, stop_requested=lambda: False) -> bool:
        """Bound local residual behind an unmerged proposal without blocking startup."""
        limit = self.settings.max_offline_steps
        clock = self.local_steps
        while any(self.steps_since[i] >= limit for i in (set(self.pending) | self.awaiting_global)):
            if stop_requested():
                return False
            clock += 1
            self._collect_uploads(clock)
            self._collect_downloads()
            time.sleep(.2)
        return True

    def resume(self, value: dict) -> None:
        if value["protocol"] != PROTOCOL or value["fingerprint"] != self.manifest["fingerprint"]:
            raise ValueError("DiLoCo learner state belongs to a different coordinator run")
        if value["worker"] != self.worker:
            raise ValueError("DiLoCo learner state belongs to another data shard")
        self.session = value["session"]
        self.versions = list(value["versions"])
        self.baselines = [None if part is None else part.float().cpu() for part in value["baselines"]]
        self.steps_since = list(value["steps_since"])
        self.tokens_since = list(value["tokens_since"])
        self.sequences = list(value["sequences"])
        self.local_steps = int(value["local_steps"])
        self.local_tokens = int(value["local_tokens"])
        self.pending = {int(k): dict(v, future=None) for k, v in value["pending"].items()}
        self.awaiting_global = set(value.get("awaiting_global", []))
        ledger = self._lease()
        self._start_heartbeat()
        for index, entry in list(self.pending.items()):
            filename = self.outbox / f"{index:05d}-{entry['sequence']:09d}.safetensors"
            if not filename.exists():
                payload = value.get("pending_payloads", {}).get(str(index))
                if payload is None:
                    raise FileNotFoundError(f"Learner state lacks pending upload {index}")
                entry["delta"] = payload
            entry["filename"] = str(filename)
            self._send_async(index)
        self._adopt_ledger(ledger)
        self._fetch_all()

    # -- fragment transfer ---------------------------------------------------

    def _fetch(self, index: int, version: int) -> tuple[int, torch.Tensor] | None:
        path, headers = self.transport.fetch_file(f"/v1/fragments/{index}?after={version}", self.outbox)
        if path is None:
            return None
        try:
            vector = load_tensors(path)["theta"]
            if (
                vector.shape != (self.fragments[index].length,)
                or vector.dtype not in (torch.bfloat16, torch.float32)
                or not bool(vector.isfinite().all())
            ):
                raise ValueError("Invalid global fragment received")
            return int(headers["X-Fragment-Version"]), vector.float()
        finally:
            path.unlink(missing_ok=True)

    def _install(self, index: int, update: tuple[int, torch.Tensor] | None) -> bool:
        """Install a newer global fragment without discarding local progress.

        ``baselines[index]`` is the anchor that local progress is measured from:
        the last installed global, or the upload point once a pseudo-gradient
        was queued. Steps trained since the anchor are carried onto the new
        global (overlapped DiLoCo), so learners never idle while a round is in
        flight, and the next pseudo-gradient contains exactly that residual.
        """
        if update is None:
            return False
        version, tensor = update
        if version <= self.versions[index]:
            return False
        anchor = self.baselines[index]
        values = tensor if anchor is None else tensor + (pack_fragment(self.surface.masters, self.fragments[index]) - anchor)
        install_fragment(self.surface, self.fragments[index], values)
        self.versions[index] = version
        self.baselines[index] = tensor
        self.awaiting_global.discard(index)
        self.installs += 1
        return True

    def _fetch_all(self) -> None:
        for index in range(len(self.fragments)):
            self._install(index, self._fetch(index, self.versions[index]))

    def _upload(self, index: int, entry: dict) -> dict:
        file = self._write_outbox(entry)
        _, response, _ = self._with_lease(
            lambda: self.transport.request(
                "POST",
                f"/v1/updates/{index}",
                file=file,
                headers={
                    "X-Worker": str(self.worker),
                    "X-Session": self.session,
                    "X-Sequence": str(entry["sequence"]),
                    "X-Base-Version": str(entry["version"]),
                    "X-Steps": str(entry["steps"]),
                    "X-Tokens": str(entry["tokens"]),
                    "Content-Type": "application/octet-stream",
                },
            )
        )
        return response

    def _send_async(self, index: int) -> None:
        entry = self.pending[index]
        entry["future"] = self.io.submit(self._upload, index, dict(entry))

    def _write_outbox(self, entry: dict) -> Path:
        filename = Path(entry["filename"])
        if not filename.exists():
            save_tensors(filename, {"delta": entry["delta"]})
        return filename

    def _flushable(self, index: int) -> bool:
        """Local progress the coordinator has not been sent; fully frozen fragments never change."""
        return (
            self.steps_since[index] > 0
            and self.tokens_since[index] > 0
            and not all(part.name in self.surface.frozen_names for part in self.fragments[index].slices)
        )

    def _queue(self, index: int) -> None:
        if index in self.pending or index in self.awaiting_global or self.versions[index] < 0:
            return
        if not self._flushable(index):
            return
        baseline = self.baselines[index]
        assert baseline is not None
        current = pack_fragment(self.surface.masters, self.fragments[index])
        delta = baseline - current
        if not torch.isfinite(delta).all():
            raise ValueError("Nonfinite local DiLoCo pseudogradient")
        sequence = self.sequences[index] + 1
        self.sequences[index] = sequence
        self.pending[index] = {
            "filename": str(self.outbox / f"{index:05d}-{sequence:09d}.safetensors"),
            "sequence": sequence,
            "version": self.versions[index],
            "steps": self.steps_since[index],
            "tokens": self.tokens_since[index],
            "delta": delta.to(torch.bfloat16 if self.settings.wire_dtype == "bf16" else torch.float32),
            "future": None,
            "retry_after": 0,
        }
        # Progress from here on is measured against the upload point, so steps
        # trained while this round is in flight survive the global install.
        self.baselines[index] = current
        self.steps_since[index] = 0
        self.tokens_since[index] = 0
        self._send_async(index)

    def _retry_later(self, entry: dict, steps: int, error: Exception) -> None:
        entry["future"] = None
        entry["retry_after"] = steps + min(64, 2 ** min(6, entry.get("attempts", 0)))
        entry["attempts"] = entry.get("attempts", 0) + 1
        self.last_error = str(error)

    def _collect_uploads(self, steps: int) -> int:
        errors = 0
        for index, entry in list(self.pending.items()):
            future = entry["future"]
            if future is None or not future.done():
                continue
            try:
                response = future.result()
                Path(entry["filename"]).unlink(missing_ok=True)
                del self.pending[index]
                self.awaiting_global.add(index)
                self.offline_steps = 0
                # Our upload completed a quorum: fetch the merge now, not at the next heartbeat.
                if isinstance(response, dict) and int(response.get("version", -1)) > self.versions[index] and index not in self.downloads:
                    self.downloads[index] = self.io.submit(self._fetch, index, self.versions[index])
            except RemoteError as exc:
                if exc.status == 409 and exc.detail.get("stale"):
                    # Keep the exact outbox and local history for explicit recovery.
                    # Dropping this update would falsely report complete exposure.
                    raise RuntimeError("DiLoCo update became stale; final contribution remains undelivered") from exc
                elif 400 <= exc.status < 500 and exc.status not in (408, 409, 425, 429) and entry.get("attempts", 0) >= 3:
                    raise RuntimeError("DiLoCo update was rejected; contribution remains undelivered") from exc
                else:
                    errors += 1
                    self._retry_later(entry, steps, exc)
            except (OSError, TimeoutError, ValueError) as exc:
                errors += 1
                self._retry_later(entry, steps, exc)
        for index, entry in list(self.pending.items()):
            if entry["future"] is None and steps >= entry["retry_after"]:
                self._send_async(index)
        return errors

    def _collect_downloads(self) -> tuple[bool, int]:
        updated = False
        errors = 0
        for index, future in list(self.downloads.items()):
            if not future.done():
                continue
            try:
                updated = self._install(index, future.result()) or updated
                self.offline_steps = 0
            except (OSError, TimeoutError, ValueError, RemoteError) as exc:
                errors += 1
                self.last_error = str(exc)
            del self.downloads[index]
        if self.remote_versions is not None:
            for index, remote in enumerate(self.remote_versions):
                if remote > self.versions[index] and index not in self.downloads:
                    self.downloads[index] = self.io.submit(self._fetch, index, self.versions[index])
        return updated, errors

    # -- training-thread hooks -----------------------------------------------

    def after_step(self, *, tokens: int, report: Mapping[str, Any] | None = None) -> dict:
        """Call after each complete optimizer update; never blocks on the network."""
        self.local_steps += 1
        steps = self.local_steps
        self.local_tokens += tokens
        if report:
            self.report = dict(report)
        for i in range(len(self.fragments)):
            self.steps_since[i] += 1
            self.tokens_since[i] += tokens
        errors = self._collect_uploads(steps)
        updated, download_errors = self._collect_downloads()
        errors += download_errors
        if self.heartbeats != self.heartbeats_seen:
            self.heartbeats_seen = self.heartbeats
            self.offline_steps = 0
        if self.heartbeat_error:
            errors += 1
            self.last_error = self.heartbeat_error
        # H local steps per fragment with stable staggered offsets; no global barrier.
        H = self.settings.local_steps
        for index in range(len(self.fragments)):
            first_due = H + index * H // len(self.fragments)
            if (
                self.local_steps >= first_due
                and self.steps_since[index] >= H
                and index not in self.pending
                and index not in self.awaiting_global
            ):
                self._queue(index)
        if errors:
            self.offline_steps += 1
        if self.offline_steps > self.settings.max_offline_steps:
            raise RuntimeError(f"DiLoCo coordinator unreachable for too many updates: {self.last_error}")
        return {"updated": int(updated), "pending": len(self.pending), "merges": self.merges, "installs": self.installs}

    def declare_completion(self, *, drained: bool) -> None:
        body={'worker':self.worker,'session':self.session,'drained':drained}
        if drained:body['sequences']=list(self.sequences)
        self._with_lease(lambda:self.transport.json('POST','/v1/completion',body))

    def finish(self, *, stop_requested=lambda: False) -> bool:
        """Send the last partial window and wait until it is delivered (bounded).

        Progress made after a round that still awaits its merge can be sent
        only once that merge installs, so this waits for it until the deadline.
        Returns false rather than claiming delivery when any local window remains unsent.
        """
        deadline = time.monotonic() + 2 * self.manifest["grace_seconds"] + self.settings.timeout_seconds
        clock = self.local_steps
        def peer_steps():
            status=self.transport.json('GET','/v1/status')
            return {worker:int(value.get('steps',0)) for worker,value in status.get('worker_progress',{}).items()
                    if worker!=str(self.worker)}
        try:peers=peer_steps()
        except (OSError,TimeoutError,ValueError,RemoteError):peers={}
        while True:
            clock += 1
            self._collect_uploads(clock)
            self._collect_downloads()
            for index in range(len(self.fragments)):
                self._queue(index)
            expired = time.monotonic() > deadline
            unsent = [i for i in range(len(self.fragments)) if self._flushable(i)]
            if not self.pending and not unsent:
                return True
            if stop_requested():
                return False
            if expired:
                try:current=peer_steps()
                except (OSError,TimeoutError,ValueError,RemoteError):current={}
                if any(steps>peers.get(worker,0) for worker,steps in current.items()):
                    peers=current
                    deadline=time.monotonic()+2*self.manifest["grace_seconds"]+self.settings.timeout_seconds
                else:
                    print("DiLoCo final round not fully delivered; peers made no observed progress", flush=True)
                    return False
            time.sleep(0.2)

    def state_dict(self) -> dict:
        """Baselines, cursors and pending payloads; never live futures or sockets."""
        return {
            "protocol": PROTOCOL,
            "fingerprint": self.manifest["fingerprint"],
            "worker": self.worker,
            "session": self.session,
            "versions": list(self.versions),
            "baselines": [None if t is None else t.detach().clone() for t in self.baselines],
            "steps_since": list(self.steps_since),
            "tokens_since": list(self.tokens_since),
            "sequences": list(self.sequences),
            "local_steps": self.local_steps,
            "local_tokens": self.local_tokens,
            "pending": {
                str(k): {key: v for key, v in entry.items() if key not in ("future", "delta")}
                for k, entry in self.pending.items()
            },
            "awaiting_global": sorted(self.awaiting_global),
            "pending_payloads": {
                str(k): entry["delta"] if "delta" in entry else load_tensors(entry["filename"])["delta"]
                for k, entry in self.pending.items()
            },
        }

    def metrics(self) -> dict:
        return {
            "diloco_worker": self.worker,
            "diloco_global_tokens": self.global_tokens,
            "diloco_local_tokens": self.local_tokens,
            "diloco_min_version": min(self.versions) if self.versions else -1,
            "diloco_pending_uploads": len(self.pending),
            "diloco_inflight_downloads": len(self.downloads),
            "diloco_merges": self.merges,
            "diloco_installs": self.installs,
            "diloco_offline_steps": self.offline_steps,
            **({"diloco_start_wait": dict(self.start_wait)} if getattr(self, "start_wait", None) else {}),
        }

    def close(self) -> None:
        self.stopping.set()
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join(timeout=self.settings.timeout_seconds + 5)
        self.io.shutdown(wait=True, cancel_futures=False)

