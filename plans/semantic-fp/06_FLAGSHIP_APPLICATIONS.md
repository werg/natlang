# WS6: Flagships — dogfooding tools first, then repeat-agentic-work apps that beat a frontier agent loop

**Parent:** [Campaign index](README.md). **v2** (2026-11-23).

v1 set the bar at "matches frontier quality at ≥ 10× lower cost". v2 raises it. For repeat agentic work, a compiled
local 1.7B natlang program should **beat** an unstructured frontier-agent loop on quality. Its repeated paths should
specialise to 0-token crisp code. And it should improve itself from its own runs. v2 also adds a tier v1 lacked: the
campaign's own tools (WS3, WS7) are the first flagships. They decide things, they run on LFM3, and the campaign
depends on them working.

## 1. Contract for every flagship
1. Written to WS4 §4's eight design items, with types first. `natlang check` is clean, and `--idiom` is clean once I2
   exists.
2. `Untrusted<T>` at every external boundary; `Is<T,P>` / `ensures:` on every checkable property; mutating services
   are exactly-once (`ONCE_EFFECTS`); verifiers are services.
3. Every policy is a natlang function or a `pluggable` with both sides; exact plumbing is TypeScript. There is no
   size rule. v1's "< 40 words per `.nl`" and "< 150-line TS root" are dropped: a judgment needs the words it needs,
   and an orchestration that decides is policy.
4. Fixture pack with train / held splits by `split_group`. The held split is reserved for held-v4 (WS5 H1) before
   any training use. The train split is an O1 task source (WS5).
5. `bench/` runs the four arms of §4 and writes `bench/results/<date>.json`.
6. `natlang improve` (WS8 N4) runs on it once, with no human edits, and the result is recorded.

## 2. Tier 0: the campaign's own tools (Wave A-B)

| Flagship | Built in | What it decides | Target (LFM3, local) |
|---|---|---|---|
| `applications/idiom-reviewer` | WS3 I2 | which sentences are mechanism vs judgment; prose interpreter; cue words; untyped external text | sentence-group class agreement with a Luna-run reviewer ≥ 0.85 on 300 labelled groups from the legacy corpora (labels from the teacher run, spot-checked by the agent). On the WS3 acceptance programs, finds every known prose-algorithm and reports nothing on the idiomatic set |
| `applications/migration` `migrateProgram` | WS7 M2 | how to decompose a prose program; keep or delete | ≥ 80 of 111 programs migrated or deliberately deleted with recorded reason; ≥ 30% of accepted migrations produced by LFM3 calls (rest by teacher fallback); every accepted one passes WS7 §4 |
| `applications/task-author` | WS3 T3 | seed, world, oracle, whether a task is genuinely semantic | ≥ 70% of authored tasks admitted; 100% of admitted tasks fail their keyword baseline; repetition within `measure_repetition.py` bounds of the existing supply |
| `escalate` nl policy | WS2 R4 | whether to escalate a decision | in shadow against crisp floors on 2 flagships: escalation rate within ±5 pts of crisp at equal or better final accuracy |

## 3. Tier 1-3: repeat agentic work

Targets compare Arm D (compiled LFM3) against Arm A (frontier unstructured agent loop) on the held split. "Beats"
means a higher verifier pass rate, paired over items, permutation p < 0.05.

| # | Flagship | Repeat work | Semantic-FP core | Quality target vs Arm A | Cost / latency target | Crisp target (steady state) | Wave |
|---|---|---|---|---|---|---|---|
| 1 | `pr-shepherd` | review PRs: hunk classes, invariant breaks, missing tests, doc drift | record `readout: decision` per hunk + `escalate`; `.in(teamStandards)`; `ensures:` quote ⊂ hunk; doc fixes through `proposeAndVerify` with `tsc`+tests | beats on planted-defect F1 over 50 seeded PRs; 0 hallucinated quotes | ≤ 1/20 cost, ≤ 1/5 p50 latency | ≥ 60% of hunks (lockfiles, formatting, imports, snapshots) served by crisp cases | B |
| 2 | `semantic-etl` | map drifting vendor feeds into a strict schema | `.nl` returning a typed row transformer (L4); `semanticMatch` for entity linking; `specialize` over row shapes | beats on exact field accuracy over 12 drift scenarios | ≤ 1/100 cost | ≥ 95% of rows crisp after `specialize`; ≥ 1,000 rows/s on those | B |
| 3 | `sem-sheet` | analytical grids mixing formulas and judgment cells | exact reactive DAG in TS; cells are inline `nl<T>` with decision readout; `pluggable` per column; runs in the page | beats on 3 × 50-row workbooks | decision cell p50 ≤ 3 s in the page (wllama, CPU) | stable columns promoted to crisp with `pluggable` shadow agreement ≥ 0.98 | B |
| 4 | `inbox-pilot` | daily mail / chat / notifications against standing rules in prose | `compileRule` per rule, validated on examples; `Is<Reply,...>`; `ONCE_EFFECTS` drafts and holds | beats on action accuracy over 200 messages; 0 injection escapes | ≤ 1/20 cost | ≥ 50% of messages crisp (newsletters, receipts) | C |
| 5 | `ci-doctor` | cluster CI failures, flake vs regression, bisect, verified fix | `semanticPartition` of traces; decision `flake\|env\|regression`; `iterateOn` bisect with NL stop; `branchAndSelect` fixes | beats on root-cause cluster purity and on fixes that turn tests green without removing assertions | ≤ 1/10 cost | known flake signatures crisp | C |
| 6 | `sem-merge` | resolve merge conflicts preserving both intents | `semanticMerge` + `semanticRepair`; `ensures:` both intents preserved; explicit unresolved markers | beats on 100 real conflicts (from `bgkit-data/repos` history), judged by the repository's own tests where they exist | ≤ 1/20 cost | non-overlapping hunks are crisp by construction | C |
| 7 | `runbook-engine` | execute Markdown runbooks with safety gates | step graph in TS; decision gates on risk floors; `ONCE_EFFECTS`; `iterateOn` health checks | beats on step correctness over 30 simulated incidents; 0 unconfirmed destructive actions | ≤ 1/20 cost | routine checks crisp | C |
| 8 | `sem-grep-sql` | cross-cutting questions over repos, docs, CSVs, SQLite | `nldb` generalised: exact joins + cached semantic predicates; `Neuralese<T>` views per file SHA | beats on 60 questions over 1,000+ files | repeated query ≤ 10 s on LFM3 | cached predicate hits ≥ 80% on repeats | D |
| 9 | `codemod-studio` | repo migrations where AST codemods cover most files | crisp codemod first; `.nl` transformers inside `semanticRepair` with `tsc`+tests; `specialize` mines new crisp rules | beats on files migrated green on 5 real migrations | ≤ 1/20 cost | mined crisp rules cover ≥ 50% of the semantic remainder on a second, similar migration | D |
| 10 | `diligence` | evaluate papers / libraries / vendors against a rubric | `Untrusted` evidence; `Neuralese<T>` views; recursive `.nl` over sections (L7); `ensures:` citations; `natlang tune` on past verdicts | beats on rubric agreement with expert labels over 40 dossiers | ≤ 1/20 cost | n/a | D |

`applications/wiki/merge`, `applications/nldb` and `applications/migration` are the existing seeds of 6, 8 and 9. The
flagships extend them rather than forking them (one implementation per concept).

## 4. Benchmark protocol
| Arm | Configuration |
|---|---|
| A | frontier unstructured agent loop: `gpt-6-luna` (openai-codex provider) in a standard tool loop with the same services and a prose task prompt |
| B | the natlang program on the same frontier model |
| C | the natlang program on base LFM3-1.7B (`5ff07638`), no compiled cases, no escalation |
| D | the natlang program on the current policy (`ops/current-policy.json`) after `natlang specialize`, `natlang tune` on the train split, and `escalate` to Arm B's model |

Reported per app: verifier pass rate (paired permutation p and bootstrap CI vs A), p50/p95 latency, tokens and
dollars per 100 units (frontier at list price, local at measured GPU-seconds), crisp-served share, local-resolved
share, escalation share, safety counters (unverified commits, injection escapes, schema violations). Every arm runs
on the held split only.

## 5. Self-improvement target
For each Tier 1-2 flagship, `natlang improve DIR` (WS8 N4) runs once from the shipped version, with no human edits.
Target: confirmation-split pass rate rises ≥ 5 pts on at least 3 of the 7, and no app gets worse (the acceptance
transaction rejects a worse proposal). Its proposals and their outcomes are also `improve-program` tasks (WS5 T1).

## 6. Work items
| ID | Item | Depends | Size |
|---|---|---|---|
| A0 | `bench/` harness shared by all flagships: arms A-D, held-split runner, results schema (`scripts/semantic_fp/bench.py` + `ts-host` runner) | none | 2 |
| A1 | pr-shepherd | S1, R4 (flat floors), L7 `ensures` (or a crisp `check` stand-in until it lands) | 4 |
| A2 | semantic-etl | L4, S1 | 4 |
| A3 | sem-sheet | S1, playground runtime | 4 |
| A4-A7 | inbox-pilot, ci-doctor, sem-merge, runbook-engine | S2, S3, L3 | 3-4 each |
| A8-A10 | sem-grep-sql, codemod-studio, diligence | WS8 N3, L7 recursion | 4 each |

Tier 0 items are owned by WS3 and WS7.
