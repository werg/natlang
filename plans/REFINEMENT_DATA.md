# Refinement-type training data

Status: generators and tests done; the teacher-dependent stages are not run yet. Design: [REFINEMENT_TYPES.md](REFINEMENT_TYPES.md)
section 5. Everything here is held: no corpus has training admission (see "Admission").

No generator loads a model. Tests use scripted models or none. Commands that call the teacher run in the teacher window
(`teacher-window.json`), through the memory ledger, with at most 4 requests in flight.

## Families

| Family | Corpus | State | Producer |
|---|---|---|---|
| `refine-judge` (c) exact labels | `refine-judge-exact-20261009-v1` (1327 rows, 93 predicates) | registered, held | `scripts/refine_data/exact.py` |
| `refine-judge` (b) near-miss | not generated | needs the teacher | `applications/refine-data/*.nl`, `ts-host/scripts/refine-data/cli.mjs` |
| `refine-judge` (a) mined | not generated | needs call stores with `refinement_check` events | `scripts/refine_data/miner.py` |
| `refine-repair` | not collected | needs live teacher runs on refined applications | `ts-host/src/teacher/refine-repair.ts` |
| `refine-author` | `refine-author-static-20261009-v1` (28 examples) | registered, held | `scripts/refine_data/author.py` |
| `refine-calibration` | no data | format and review tool only | `scripts/refine_data/calibration.py` |

Entry point for all Python steps: `python3 scripts/refine_data_cli.py --help`. Code: `scripts/refine_data/`. Tests:
`tests/test_refine_data_*.py`, `ts-host/test/refine-data.test.mjs`, `ts-host/test/refine-repair.test.mjs`.

## refine-judge

**Row format.** `natlang.decision-prompt/1`, the format of `ts-host/scripts/skills/export-decision-prompts.mjs` that
`natlang_neuralese.train.decision` reads (`--prompts`, `--target gold|teacher:NAME|mix:NAME:W`). A row is one judge call:

- `messages`: the exact judge prompt of the runtime (`decisionJudge`, `ts-host/src/native/refinement.ts`). The Python
  mirror (`common.judge_messages`) is parity-tested against the TypeScript judge.
- `options`: `["true", "false"]`; `gold`: `[P(true), P(false)]`; `teachers`: `{teacher id: [P(true), P(false)]}`;
  `kind` `noul`; `family`, `role` (`train` or `heldout`), `group` (the predicate id).
- Extra fields the trainer ignores: `split`, `predicate`, `predicate_id`, `value_sha256`, `source`, `label_source`, and per
  source `edit`, `pair_id`, `observed`, `shadow_disagreement`.

The trainer's held-out rows are the ones with `role: heldout`, reported per `family`. Families are named
`refine-judge/exact/<kind>`, `refine-judge/near-miss`, `refine-judge/mined`, `refine-judge/calibration`.

**Split rule (`refine-judge-split/1`, recorded in every manifest).** The predicate text, whitespace-normalised, goes to
bucket `int(sha256("refine-judge-split/1|" + text)[:8], 16) % 10`. Buckets 0 and 1 are held out (`heldout-predicate`).
Every row of a predicate has the same split whatever its source or value, so the same predicate never trains in one source
and tests in another. The exact corpus also holds out four families whole (`clock-time`, `ipv4`, `multiple-of`,
`sorted-list`; `heldout-family`) to test transfer to a new kind of predicate. The near-miss and mined rows use the
predicate bucket only.

**Sources.**

- (c) Exact: 93 predicate texts with crisp checkers (length, format, enumeration, numeric range, list shape), 6 of them the
  applications' own predicates. A row's label is its checker's verdict; a generated value whose verdict disagrees with its
  intent is dropped and counted (0 in the registered corpus). Run: `refine_data_cli.py exact`.
- (b) Near-miss: `exemplify.nl` writes satisfying values for the 117 harvested predicates (from the "Refinements" tables of
  the `DECOMPOSITION.md` of games, wiki, logs, build, migration, scheduling and workflow); `nearMiss.nl` edits each into a
  minimal violation; `verifyNearMiss.nl` is the independent second pass. A pair is accepted only if the edit differs, keeps
  the value's type, and the verifier says original holds, edit fails, and the edit is minimal. The teacher judge then scores both sides; a
  pair on which its P(true) disagrees with the construction (original below 0.5 or edit at least 0.5) is written to
  `disagreements.jsonl` and not trained on. In accepted rows `gold` is the construction (1 and 0) and `teachers` the soft
  teacher distribution.
- (a) Mined: `miner.py` reads `refinement_check` and `refinement_shadow` events from call stores (`calls.sqlite` plus
  blobs) and trace JSONL files and emits unlabeled pairs. The checker's own verdict is kept as `observed` but is not the
  label. Previews cut at 400 characters are completed from the store when the recorded output hashes to `value_sha256`,
  else counted as unrecoverable. Shadow disagreements (crisp against judge) are flagged as hard negatives. The teacher
  judge labels them; `gold` is its distribution.

**Teacher labels.** The scorer is the teacher's `decide` (`prompt_logprobs`, `openai-compatible.ts`), run through the
runtime's own `decisionJudge`, so the prompt scored is the prompt trained on. Each label is stored with its teacher id.

## refine-repair

`ts-host/src/teacher/refine-repair.ts`. `collectRefineRepairs(row, {trace})` reads the `trajectory` frames of a collector row
(`natlang.teacher_trajectory.native/1`, frames from `trajectoryTurn`) and finds episodes: a `return_result` (or a text reply)
rejected with `refinement-unsatisfied` or `refinement-undecided` in the next context, then an attempt that stood. The record
(`natlang.refine-repair/1`) keeps every turn of the invocation up to the repair as `messages` / `tools` / `target`, like the
teacher turn records, plus the rejections (code, path, predicate, probability, feedback text, attempted value) and the repair.
Weights follow `trajectories.py` (`--context-weight 1.0`, `--feedback-weight 0.25`): instructions and inputs weigh 1,
the rejection text and other tool results after the first reply weigh 0.25, earlier assistant replies weigh 0 in later
prompts because their own turn records supervise them. The rejected attempt keeps its place in the prompt of the repair
turn but its own target has `target_weight` 0 and denied admission, like every failed proposal in the native
materializer. Rows the collector rejected (`outcome.accepted` not true) yield nothing; with `--trace`, a repair must be
confirmed by a failing then passing `refinement_check`. `recordTurns(driver)` captures frames during a live run without the
collector. Flattening a record into the standard turn records is a pure mapping of `turns`; it is not written yet because no
trainer admits these records.

## refine-author

`applications/refine-data/authoring.json` anchors each example in the repository: the exact guard text in a `.nl` body,
the slot of the `DECOMPOSITION.md` table that states the same property, and what stays readable after the guard is cut.
`author.py` reads the function and `types.ts` from the working tree (24 examples) or from git history (4: `logs/assess.nl`
at `68cb32f5^`, `wiki/reconcile.nl` at `b8d422e2^`, `build/choose.nl` at `75323731^`, `migration/propose.nl` at `da2b6519^`),
removes the guard text, refines the field in `types.ts` (or the `returns:` line) with the table's predicate and base type, and
emits the pair with unified diffs and a user/assistant `messages` form (`natlang.refine-author/1`). An entry whose text, slot or field is
missing is an omission with its reason (none today); `report.uncovered_predicate_ids` in the manifest lists the 92 of 117
harvested predicates with no example yet (their constraints are not stated as guard sentences in an instruction, or are
checked by host code only). `ts-host/scripts/refine-data/check-author.mjs` parses every refined type with the runtime parser.
Not compiled as whole projects. Hold out by app, not by example.

## refine-calibration

Format: `queue.jsonl` (`natlang.refine-calibration-item/1`, no model probabilities, so a reviewer is not anchored) and
append-only `verdicts.jsonl` (`natlang.refine-calibration/1`: `verdict` true, false or `"unclear"`, reviewer, time, note,
round; the latest verdict of an item wins). Tool, all in `refine_data_cli.py calibration --action ...`:

```sh
python3 scripts/refine_data_cli.py calibration --action queue --rows ROWS.jsonl --per-predicate 4 --out queue.jsonl
python3 scripts/refine_data_cli.py calibration --action review --queue queue.jsonl --verdicts verdicts.jsonl --reviewer NAME
python3 scripts/refine_data_cli.py calibration --action status --queue queue.jsonl --verdicts verdicts.jsonl
python3 scripts/refine_data_cli.py calibration --action export --verdicts verdicts.jsonl --out calibration-rows.jsonl
python3 scripts/refine_data_cli.py calibration --action report --verdicts verdicts.jsonl --predictions PRED.jsonl
```

`review` is interactive (t/f/u/s/q) and resumes where it stopped. `export` gives evaluation rows (`role heldout`, split
`calibration`); they never enter a training corpus. `report` prints Brier and ECE of `{id, p_true}` predictions against the
clear verdicts, overall and per family.

## Commands for the teacher window

Run on the DGX in `/home/werg/natlang` after the window opens (`runs/research-teacher-20261006/teacher-window.json`). Check
the ledger first (`python3 scripts/memory_ledger.py status`); each unit needs about 2 GB. Find the served model id with
`curl -s http://127.0.0.1:8082/v1/models` and set `MODEL`. Teacher requests are limited to 4 in flight (`--concurrency 4`).
Stage outputs resume by id, so an interrupted unit is simply started again. Work in `runs/refine-data-20261009/`.

```sh
cd /home/werg/natlang && git pull --rebase origin main && (cd ts-host && npm run build:node)   # under the ledger too
W=runs/refine-data-20261009; mkdir -p $W; MODEL=...   # the id from /v1/models
L() { unit=$1; shift; python3 scripts/memory_ledger.py run --unit natlang-refine-$unit --budget-gb 2 --class experiment --wait 3600 \
  --workdir /home/werg/natlang -- sh -c "PATH=/home/werg/.local/bin:\$PATH; $* > $W/$unit.log 2>&1"; }
T="--endpoint http://127.0.0.1:8082 --model $MODEL --concurrency 4"

# 0. no model: seeds, request files
python3 scripts/refine_data_cli.py exemplify-requests --out $W/ex-req.jsonl --count 6

# 1. satisfying values for the harvested predicates (recorded calls: NATLANG_CALL_STORE defaults to the machine store)
L exemplify "node ts-host/scripts/refine-data/cli.mjs stage --stage exemplify --in $W/ex-req.jsonl --out $W/ex-res.jsonl $T"
python3 scripts/refine_data_cli.py ingest-exemplify --requests $W/ex-req.jsonl --results $W/ex-res.jsonl --out $W/satisfying.jsonl

# 2. minimal violations, then the independent verification pass
python3 scripts/refine_data_cli.py near-miss-requests --satisfying $W/satisfying.jsonl --out $W/nm-req.jsonl
L nearmiss "node ts-host/scripts/refine-data/cli.mjs stage --stage nearMiss --in $W/nm-req.jsonl --out $W/nm-res.jsonl $T"
python3 scripts/refine_data_cli.py verify-requests --satisfying $W/satisfying.jsonl --near-miss-results $W/nm-res.jsonl --out $W/vf-req.jsonl
L verify "node ts-host/scripts/refine-data/cli.mjs stage --stage verifyNearMiss --in $W/vf-req.jsonl --out $W/vf-res.jsonl $T"
python3 scripts/refine_data_cli.py accept --satisfying $W/satisfying.jsonl --near-miss-results $W/nm-res.jsonl --verify-results $W/vf-res.jsonl --out $W/accepted.jsonl

# 3. mined pairs (no model); stores and traces are those of the refined-application runs
python3 scripts/refine_data_cli.py mine --store ~/.local/share/natlang/calls --trace PATH.jsonl --out $W/mined.jsonl

# 4. teacher judge labels for both sides of every accepted pair and every mined pair (the scorer is prompt_logprobs)
python3 scripts/refine_data_cli.py label-requests --accepted $W/accepted.jsonl --mined $W/mined.jsonl --out $W/label-req.jsonl
L label "node ts-host/scripts/refine-data/cli.mjs label --in $W/label-req.jsonl --out $W/labels.jsonl --teacher teacher:$MODEL $T"

# 5. assemble (no model); then publish under the registry conventions, held
python3 scripts/refine_data_cli.py assemble --out-dir data/neuralese/corpora/refine-judge-teacher-20261009-v1 --id refine-judge-teacher-20261009-v1 \
  --teacher teacher:$MODEL --accepted $W/accepted.jsonl --mined $W/mined.jsonl --labels $W/labels.jsonl
# add a registry entry (copy the refine-judge-exact one, admission held, training_admission false), then:
python3 scripts/sync_training_corpora.py publish --machine dgx --id refine-judge-teacher-20261009-v1

# refine-repair: after live teacher runs on refined applications (collector rows, optionally with the run's trace events)
L repairs "node ts-host/scripts/refine-data/collect-repairs.mjs --in runs/teacher.native.jsonl --out $W/repairs.jsonl"
```

If the teacher's `decide` fails with `decision-unsupported`, the server returns no `prompt_logprobs`; stop and report, since
labels from another source are a separate decision. If the teacher thinks before replying, pass
`--request '{"chat_template_kwargs":{"enable_thinking":false}}'` to `label` so the scored continuation is the bare reply.
Before the near-miss unit, spot-check 20 `ex-res` values by hand: if the exemplars do not satisfy their predicates, the
near-miss pairs inherit that.

## Admission criteria (proposed; the decision is the owner's)

Copies, availability and a clean schema check do not grant admission. Every family needs these explicit, recorded in the
registry entry (`admission`, `training_admission`) with the evidence:

1. **Exact (`refine-judge-exact-20261009-v1`).** Admit as format and calibration-at-the-extremes data only after a student
   trained on its `train` split beats the untrained readout on `heldout-predicate` and `heldout-family` Brier, and loses no
   accuracy on the near-miss held-out rows. It is not evidence of ability on the hard predicates.
2. **Near-miss.** A hand check of 100 random accepted pairs: at least 90 where the original satisfies, the edit does not, and the
   edit is minimal. Verifier-rejected and teacher-disagreement counts reported per predicate; predicates with an
   acceptance rate under 20% dropped, listed. No leakage: the predicate bucket of each row matches the split rule.
3. **Mined.** Values that came from user text or private repositories reviewed for admissibility; the teacher's label
   distribution reported against the checker's observed verdicts; shadow disagreements reviewed as possible crisp-checker bugs.
4. **Repair.** At least 200 episodes across at least 5 applications, repaired values hand-checked on 50 against the
   predicate, and the trainer's `--feedback-weight` confirmed to apply to the rejection text. Rejected attempts stay out of
   positive targets.
5. **Author.** Review of the 28 edited instructions for readability once the guard is cut; keep out any example whose
   predicate the reviewer would not write (the table is a proposal). Hold out by app when measuring.
6. **Calibration.** At least 20 reviewed items per predicate family, two reviewers on a 20% overlap with agreement reported.
   The gate on the default thresholds is Brier and ECE on this set: propose ECE at most 0.1 for the student judge, and where
   it is worse a per-predicate threshold or an escalation band (REFINEMENT_TYPES.md section 3).

Never mutate a published id: a changed corpus is a new id with the relation to the old one stated in `derived_from`.
