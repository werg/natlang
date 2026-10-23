# Memetic teacher ferry (2026-10-23)

This is local implementation only; it does not call either sandbox or generate
teacher output. Root owns deployment and start/stop operations.

The evaluator's local HTTP service exposes only `GET /health` and ordinary
`POST /v1/chat/completions` on `127.0.0.1:8766`. It checks the active native
student identity at `127.0.0.1:8765/v1/neuralese/info`, stores exact request
bytes in a synchronous SQLite WAL, and derives a deterministic request ID from
the model and raw body. A client may send `Idempotency-Key`; if absent, the
server derives it. A mismatched key/body pair is rejected. The task harness must
use `--author-ferry http://127.0.0.1:8766`; the direct teacher URL remains only
the independently bound logical author endpoint in the source receipt.

The host SDK process is `scripts/coreweave/memetic_teacher_ferry.py`. It opens
only the two already-authorized sandboxes (teacher
`78530f71-08fc-4f57-ace9-7fbad64aeeb8`, evaluator
`393f9b75-2666-456c-a850-1b46a8b7d147`). It runs fixed ferry-module operations,
checks that both deployed module files match the reviewed local module bytes,
and moves only ID-addressed request/result files with the SDK. It accepts no
endpoint, remote command, or sandbox ID from CLI input. Run continuously on the
supervising host with:

```sh
python3 scripts/coreweave/memetic_teacher_ferry.py --interval 2
```

For an evaluator-side service, copy the exact
`training/neuralese/natlang_neuralese/rl/memetic_teacher_ferry.py` bytes to
`/work/native-production/control/memetic-teacher-ferry.py` on the evaluator and
to `/work/generation/qwen36-35b-teacher-20261022/memetic-teacher-ferry.py` on
the teacher. The host refuses different deployed bytes. Provision the same
random key (at least 32 bytes, mode `0600`) at
`/run/secrets/natlang-memetic-teacher-ferry` on both sandboxes; the host never
reads it. Create writable roots at `/work/native-production/control/memetic-teacher-ferry`
and `/work/generation/qwen36-35b-teacher-20261022/memetic-teacher-ferry`.
The evaluator service command is:

```sh
python3 /work/native-production/control/memetic-teacher-ferry.py evaluator-serve --host 127.0.0.1 --port 8766
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

Pinned teacher source files are declared in the module:

- `/work/generation/qwen36-35b-teacher-20261022/actor-source-tools-v2.json`
- `/work/generation/qwen36-35b-teacher-20261022/teacher-execution-identity-tools-v2.json`
- `/work/generation/qwen36-35b-teacher-20261022/download-verified.json`

The declared sandbox, model, revision and source hashes are checked by the
teacher before authoring. The evaluator is pinned to its exact checkpoint,
tokenizer fingerprint, embedding hash and architecture. `--author-model` and
`--author-provenance` must name this same Qwen source receipt; the receipt
contains those three exact `source_bindings` entries plus its checkpoint,
tokenizer and runtime hashes. This preserves policy in the natural-language
prompt while the ferry enforces only identity and byte custody.

CPU-only fake SDK verification is in
`tests/neuralese/test_memetic_teacher_ferry.py`: SDK timeout/restart reuses one
request and results in one generation, forged HMAC delivery is refused by the
evaluator, and request IDs cannot be rebound to changed bytes.
