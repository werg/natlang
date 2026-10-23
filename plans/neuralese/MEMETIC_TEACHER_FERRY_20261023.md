# Memetic teacher ferry (2026-10-23)

This is local implementation only; it does not call either sandbox or generate
teacher output. Root owns deployment and start/stop operations.

The evaluator's local HTTP service exposes only `GET /health` and ordinary
`POST /v1/chat/completions` on `127.0.0.1:8766`. It checks the active native
student identity at `127.0.0.1:8765/v1/neuralese/info`, stores exact request
bytes in a synchronous SQLite WAL, and derives a deterministic request ID from
the serving alias, raw body and exact evaluator sandbox/checkpoint identity. A client may send `Idempotency-Key`; if absent, the
server derives it. A mismatched key/body pair is rejected. The task harness must
use `--author-ferry http://127.0.0.1:8766`; the direct teacher URL remains only
the independently bound logical author endpoint in the source receipt.

The host SDK process is `scripts/coreweave/memetic_teacher_ferry.py`. It opens
only the two already-authorized sandboxes (teacher
`78530f71-08fc-4f57-ace9-7fbad64aeeb8`, evaluator
`393f9b75-2666-456c-a850-1b46a8b7d147` for LFM or
`ee341c7b-a425-4b4b-ad7e-f3cf7cbc2369` for Mellum). Mellum remains unprovisioned
until its current native identity is supplied in the deployment manifest. The
pump accepts `--student lfm|mellum` and only opens the allowlisted ID for that
role. It runs fixed ferry-module operations,
checks that both deployed module files match the reviewed local module bytes,
and moves only ID-addressed request/result files with the SDK. It accepts no
endpoint, remote command, or sandbox ID from CLI input. Run continuously on the
supervising host with:

```sh
python3 scripts/coreweave/memetic_teacher_ferry.py --student lfm --interval 2
```

For an evaluator-side service, copy the exact
`training/neuralese/natlang_neuralese/rl/memetic_teacher_ferry.py` bytes to
`/work/native-production/control/memetic-teacher-ferry.py` on the evaluator and
to `/work/generation/qwen36-35b-teacher-20261022/memetic-teacher-ferry.py` on
the teacher. Also copy `training/neuralese/natlang_neuralese/rl/memetic_teacher_ferry_deployments.json`
byte-for-byte beside both deployed modules under the same filename. It pins the
teacher and role allowlist, contains the exact LFM native checkpoint identity,
and leaves the Mellum identity null until an operator binds its current actor.
The host refuses different deployed code or manifest bytes. Provision the same
random key (at least 32 bytes, mode `0600`) at
`/run/secrets/natlang-memetic-teacher-ferry` on both sandboxes; the host never
reads it. Create writable roots at `/work/native-production/control/memetic-teacher-ferry`
and `/work/generation/qwen36-35b-teacher-20261022/memetic-teacher-ferry`.
The evaluator service command is:

```sh
python3 /work/native-production/control/memetic-teacher-ferry.py evaluator-serve --student lfm --host 127.0.0.1 --port 8766
```

The teacher module starts one durable worker for a newly ferried request. Its
SQLite submit marker is committed before the only POST to `127.0.0.1:8003`.
If the worker dies after that marker, the request becomes terminal
`indeterminate`; it is never generated again. A terminal response includes
exact request/response hashes, evaluator checkpoint identity, teacher sandbox,
model/revision, source-file hashes, and the active vLLM listener PID, process
start ticks, and command-line hash. The evaluator verifies the HMAC before
accepting bytes. A transport timeout leaves the same durable ID for later
polling; it does not create a new generation request.

The download receipt has exactly `at`, `members`, `model`, `passed`, `repo`, and
`revision`: the local model path is
`/work/generation/qwen36-35b-teacher-20261022/model`, while `repo` is the
selected NVFP4 source `nvidia/Qwen3.6-35B-A3B-NVFP4`. The serving alias observed
from vLLM `/v1/models` is `qwen3.6-35b-a3b`; it is checked separately from the
source repository ID. The ferry binds `members` by hashing its exact canonical
value and verifies `passed`, repository, path and revision.

Pinned teacher source files are declared in the deployment manifest:

- `/work/generation/qwen36-35b-teacher-20261022/actor-source-tools-v2.json`
- `/work/generation/qwen36-35b-teacher-20261022/teacher-execution-identity-tools-v2.json`
- `/work/generation/qwen36-35b-teacher-20261022/download-verified.json`
- `/work/generation/qwen36-35b-teacher-20261022/tool-serving-check-v3.json`
- `/work/generation/qwen36-35b-teacher-20261022/server-launch-tools-v2.json`

Each SDK write is at most 24 KiB. Requests and responses are transferred as
ID-addressed chunks, then reassembled only after the exact aggregate length and
SHA-256 verify; the assembled file is atomically installed. Request and response
payloads have hard transfer bounds.

The declared teacher sandbox, source repository, serving alias, revision and
source hashes are checked before authoring. The evaluator is pinned to its exact checkpoint,
tokenizer fingerprint, embedding hash and architecture. `--author-model` and
`--author-provenance` must name this same Qwen source receipt; the receipt
contains the exact `source_bindings` entries above plus its checkpoint,
tokenizer and runtime hashes. This preserves policy in the natural-language
prompt while the ferry enforces only identity and byte custody.

CPU-only fake SDK verification is in
`tests/neuralese/test_memetic_teacher_ferry.py`: SDK timeout/restart reuses one
request and results in one generation even with 300 KiB prompts and 400 KiB
responses, enforces the 24 KiB SDK write bound, rejects incomplete or hash-mismatched
chunk assemblies, refuses forged HMAC delivery, and prevents request ID rebinding.
