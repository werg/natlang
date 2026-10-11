# Direct recurrence parent proof packet

`scripts/prepare_direct_recurrence_parent_proof.py` builds the legacy parent
proof packet for one explicitly selected recurrence checkpoint. It reads the
checkpoint and the pinned stage files, hashes them, checks the checkpoint
identity, verifies the parent manifest/payload, and writes only the requested
new proof JSON. It never copies or edits a checkpoint. The required
`--expected-checkpoint-sha256` must come from the operator's independently
selected safe-checkpoint receipt; the script refuses a different hash and
rechecks the file after reading its identity.
Stage-plan data inputs are also re-hashed before a proof is written. A relocated
parent artifact is resolved explicitly and checked against the pinned manifest
content hash.

For the V8 native E2E epoch v2, the bindings are:

| Role | Pinned path |
| --- | --- |
| Stage plan | `/srv/storage/natlang-artifacts/luna-neuralese-direct-lfm-base-20261011-v8/native-e2e-full-endpoint-epoch-v2/direct-stage-plan.json` |
| Launch intent | `/srv/storage/natlang-artifacts/luna-neuralese-direct-lfm-base-20261011-v8/native-e2e-full-endpoint-epoch-launch-v2/stage-launch-intent.json` |
| Executed launch receipt | `/srv/storage/natlang-artifacts/luna-neuralese-direct-lfm-base-20261011-v8/native-e2e-full-endpoint-epoch-launch-v2/launch.json` |
| Recipe | `/srv/storage/natlang-artifacts/luna-neuralese-direct-lfm-base-20261011-v8/native-e2e-full-endpoint-epoch-launch-v2/recipe.json` |
| Parent artifact | `/srv/storage/natlang-artifacts/luna-neuralese-direct-lfm-base-20261011-v8/direct-export-v2` |

After the owner selects a stable checkpoint and supplies its verified SHA,
prepare the packet with a new output path:

```sh
python3 scripts/prepare_direct_recurrence_parent_proof.py \
  --stage-plan /srv/storage/natlang-artifacts/luna-neuralese-direct-lfm-base-20261011-v8/native-e2e-full-endpoint-epoch-v2/direct-stage-plan.json \
  --launch-intent /srv/storage/natlang-artifacts/luna-neuralese-direct-lfm-base-20261011-v8/native-e2e-full-endpoint-epoch-launch-v2/stage-launch-intent.json \
  --launch-receipt /srv/storage/natlang-artifacts/luna-neuralese-direct-lfm-base-20261011-v8/native-e2e-full-endpoint-epoch-launch-v2/launch.json \
  --recipe /srv/storage/natlang-artifacts/luna-neuralese-direct-lfm-base-20261011-v8/native-e2e-full-endpoint-epoch-launch-v2/recipe.json \
  --checkpoint /srv/storage/natlang-artifacts/luna-neuralese-direct-lfm-base-20261011-v8/native-e2e-full-endpoint-epoch-v2/native_e2e/checkpoint.pt \
  --expected-checkpoint-sha256 <owner-supplied-safe-checkpoint-sha256> \
  --parent-artifact /srv/storage/natlang-artifacts/luna-neuralese-direct-lfm-base-20261011-v8/direct-export-v2 \
  --out /srv/storage/natlang-artifacts/luna-neuralese-direct-lfm-base-20261011-v8/native-e2e-full-endpoint-epoch-v2/parent-proof-packet.json
```

The command above is a template, not a proof receipt. Do not run it against a
checkpoint that is still being updated. The generated packet is useful only
while the referenced checkpoint and all pinned source files still match their
recorded hashes. It does not qualify the model or grant data admission.

## Execution environment

The Pop host's `python3` and `.venv` do not provide PyTorch. Run the builder
inside the pinned training container image from the stage plan, with the current
canonical checkout mounted read-only and the artifact root mounted read-write.
Do not use the frozen package snapshot: it predates the proof-verification
helpers. This is a CPU-only provenance operation; it does not load the model or
need GPU access. A command template is:

```sh
docker run --rm --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=1g \
  -v /home/werg/natlang:/home/werg/natlang:ro \
  -v /srv/storage:/srv/storage:rw \
  -e PYTHONPATH=/home/werg/natlang/training/neuralese \
  -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint /usr/bin/python3 \
  sha256:6b337ae8eb936191c4cba64641aeb0dfcb0658ae3fadb8ede6be03c8a6ccb3de \
  /home/werg/natlang/scripts/prepare_direct_recurrence_parent_proof.py \
  <the verified arguments above>
```

The actual proof-generation command remains for the owner to run after choosing
and independently pinning a stable checkpoint. These instructions and the
builder have been prepared but not exercised against the active checkpoint.
