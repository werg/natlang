# Modern static adapters and quality audit — 2026-09-28

## Admission policy

Conversions are candidates, not training approval. Preserve raw annotations, licenses,
source revisions, grouping and original holdouts. Hold ambiguous labels, missing joins,
unsupported semantics and unresolved units. Successful execution cannot promote a
held source: `source-conversion.ts` checks the source quality attestation for ordinary
teacher results as well as offline replay. Recovered sources require that attestation.

Synthetic action notes are context only, never generated reasoning. Scripted leaf
judgments remain held in the default reasoning lane. The independent, named judgment
functions in decision/state reducers provide delegation examples without capturing
all collection labels in the child opening. Existing direct/delegated teacher policy
and the three-layer ad hoc limit are unchanged.

## Implemented adapters

| Source | Conversion and quality checks | Default status |
| --- | --- | --- |
| NanoJev | Typed boolean/choice/score leaves and four-item map/count reducers; independent contract implementation; exact instruction and rubric checks | Verified supported contracts eligible; other states/instructions held |
| Typed Decisions / sales | Same modern leaf format; preserved labels, confidence and company groups; consolidated sales file only | All synthetic/teacher labels held pending independent review; HTTP errors excluded |
| FinQA | Original program parser with explicit literal/reference operands; operator/arity/reference/domain checks; exact executable answer agreement; full report evidence | **All held** pending question-unit/display-answer review; CC-BY-4.0 attribution |
| SCONE | Ordered typed transitions; joins original TSV and state chains; independent conservative drain oracle | Eight train sequences eligible; scene, tangram and other actions held |
| SGD | Service-local state reducers; raw dialogue frame and schema joins; all intervening turns retained | Single explicitly stated slot values eligible; normalization, inferred defaults and alternative values held |
| CLEVR | Symbolic typed scene programs; joins original questions/scenes; exact functional gold; singularity and all dependencies checked | Verified train candidates eligible; validation preserved |
| Synthetic programs | Modern types, arguments, projects, nested functions and fold steps; original operations/oracles retained as private metadata | Independently checked crisp formulas eligible; semantic/effect programs held |
| Leaf bank | All 620 entries joined to unique modern function definitions, including fold-step codebases | All held pending current semantic review |

The conversion library has **17,877 eligible train candidates**, **3,244 eligible
holdout candidates**, and **65,395 held conversions**. Counts overlap derivative
programs (leaves/batches) and are not distinct approved training cases. Separate raw
conversion/replay rejection ledgers preserve everything filtered out. Two train
conversions were held for source/input overlap with holdouts.

The bounded native pilot admits **72 cases / 325 decisions**: NanoJev 16, SCONE 8,
SGD 16, CLEVR 16, crisp synthetic 16. Another **229 decisions remain held**.
`data/teacher/recovered/static.manifest.json` includes only admitted pilot records;
large converted train/test/held libraries are **not** automatically trained.
Both the shell SFT builder and staged recipe discover this bundle alongside the
51-case directory bundle. Explicit static manifests can be supplied repeatedly
(recipe) or comma-separated (`NATLANG_STATIC_BUNDLE`); `off`/`--no-static-bundle`
still excludes them. No training or generation-model calls were made by this work.

## Audit of previously integrated data

- Rebuilt all 51 directory cases from the content-addressed source cache. The IR
  checksum matches the integrated bundle exactly. Fresh replay, source-specific gold,
  heldout-seed, successful-trace/scope and pinned-license checks pass. 157 decisions
  eligible, 32 direct scripted decisions held. Rebuilding preserves rejection reasons.
- Replayed all 2,192 old reference trajectories through current tools. All remain
  trajectory-admitted: 5,174 eligible decisions, 5,054 direct/failed/checker decisions
  held. This establishes replay fidelity; semantic judgments retain their existing
  constructed/source oracle qualifications.
- Sharp tokenizer audits: all 325 recovery decisions fit 8,192 tokens (max 4,593);
  all 5,174 old reference decisions fit (max 4,925). The latter include **661 excess
  rendered pairs**, discovered after rendering. The final audit now removes duplicate
  pairs, preserves their attribution, and holds train rows connected to a holdout
  through groups/programs or duplicate pairs. It retains the most restrictive
  reasoning mask for identical pairs. Final filtered reference yield is **4,513**.
- The combined static audit retains **4,995 unique decisions** from 5,656 rendered
  candidates, removes 661 duplicate pairs, finds zero train/holdout links in that
  mix, and passes max-len 8192 (maximum 6,356). Artifact:
  `runs/integrated-quality-audit-20260928/joint/sharp.ready.jsonl`.
- Audit evidence labels now distinguish scripted native references and converted
  source programs/operations from generated teacher trajectories.

Authoritative artifacts: `runs/integrated-quality-audit-20260928/` and
`runs/modern-adapters-20260928/`. Final token outputs are
`verified/sharp.filtered-ready.jsonl` in the first directory and
`verified-published/sharp.filtered-ready.jsonl` in the second. Raw rendered evidence
and rejection ledgers remain available. These are static-source audits, not a claim
that every historical Bonsai/Luna collection or future assembled mixture is certified.
The staged recipe audits every assembled mixture. The older shell builder emits rendered
evidence (`sft.jsonl`); it is not a final readiness certificate. Use the staged final
audit before training that output.

## Reproduction

```bash
python3 scripts/audit_recovered_sources.py --out runs/new-recovery/joins
node --max-old-space-size=2200 ts-host/scripts/static-adapters/build.mjs \
  --source data/external_pilot --joins runs/new-recovery/joins \
  --out runs/new-recovery/bundle --pilot 16
node ts-host/scripts/inline-curriculum/static-bundle-input.mjs \
  runs/new-recovery/bundle/static.manifest.json --turns-out runs/new-recovery/turns.jsonl
node --max-old-space-size=1600 ts-host/scripts/static-adapters/audit-integrated.mjs \
  runs/new-reference-audit runs/inline-curriculum/ref-v1.results.jsonl \
  runs/inline-curriculum/ref-composed-v1.results.jsonl
```

Use the existing CPU renderer/auditor with the exact Sharp tokenizer/template and
max-len 8192; tokenizer-only artifacts need no model weights/GPU. Builders preserve
prior evidence by requiring new output files. Source-join manifests hash the raw
archives, dialogue/schema files and old IR. Conversion checks those hashes again;
source and adapter hashes are checked before publishing the final manifest.

## Course changes and limitations

- Replaced ambiguous legacy FinQA JSON numeric operands with the original textual
  program parser. Exact-number disagreement is filtered, not silently rounded.
  Executable/display conventions still prevent any FinQA training admission.
- SCONE scene transitions appear shifted relative to utterances; no guessed repair.
  Only the independently understood drain subset is admitted.
- Early support-route adapter used `docs` instead of the source's `documentation`.
  Corrected the adapter and rubric mappings, added a regression check. The three
  apparent label discrepancies were adapter defects, **not faulty source labels**.
- Expanded legacy-definition recovery to fold steps, restoring all 620 leaf joins.
  Old review history does not authorize blanket modern semantic-label admission.
- Final corpus audit previously only reported duplicates. It now filters them and
  train/holdout links. This changes final row counts; downstream users must rebuild
  ready artifacts with the new audit, rather than reuse old audit caches.
- Legacy teacher archives, SQL disagreements and benchmark-only jeff tasks remain
  held/research evidence. No modern reasoning is fabricated for retired tools.
- Verification: Node build, browser bundle/types, 53 focused Node tests and 42 focused
  Python tests. The broader old `test_training_recipe.py` still has 14 fixture/setup
  failures from trying to run a nonexistent registry under a temporary fake repo;
  relevant recipe behavior is exercised in the supported mocked-registry tests.
- Bonsai and the single Luna worker continued independently on immutable runtime-v10.
  No queues, live runtime, worker counts or training settings were changed.
