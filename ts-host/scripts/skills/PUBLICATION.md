# Skill authoring export publication

`export-training.mjs` writes an immutable training candidate. It does not register the rows in the general training corpus.

Before proposing publication, place the positive export and source review receipt under the repository. The receipt uses `natlang.skill-authoring-source-review/1` and must bind `candidate_manifest_sha256` to the export's `manifest.json`. It must identify the reviewer, set `decision` and `source_policy.decision` to `approve` and `approved`, and list repository-relative source evidence paths with SHA-256 digests. The helper verifies those digests; the evidence still needs a human policy judgment.

Run:

```sh
node ts-host/scripts/skills/stage-training-publication.mjs \
  --repo . \
  --export runs/skill-export/positive-candidate \
  --review runs/skill-export/source-review.json \
  --out runs/skill-export/publication-proposal.json
```

The helper checks positive row count, row and sidecar hashes, support-train row family, exact paired replay for every case contributing rows, provider-free SFT, and zero DPO pairs. Other episodes may remain quarantined or missing; their attempts stay in the negative sidecar and contribute no training rows. It emits a proposal containing the exact artifact and verification hashes. It never changes `current-manifest.json`; a separate independent review must decide whether to add the proposed entry.

Zero-row and incomplete exports remain negative evidence and cannot be staged. Query, transfer, ablation, executor, and sealed evaluation data do not belong in SFT. The proposal records the negative sidecar hash for audit while pointing the training registry only at `verified-turns.jsonl`.
