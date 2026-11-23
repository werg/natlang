# RL Task Authoring Kit v2 — Typed Semantic Functional Programming

**Supersedes** sections 2–17 of `RL_TASK_AUTHORING_KIT.md` (the R1–R23 operations-desk kit) for all new Semantic-FP
batches (`sfp4+` and `held-v4` v3). Keep `RL_TASK_AUTHORING_KIT.md` only for maintaining historical batches.

## 1. The One Principle

Every authored task trains LFM3-1.7B to **execute real typed Semantic Functional Programming programs** — not to act as
a prose interpreter of deterministic TypeScript recipes, and not to match surface cue words in toy office desks.

- **TypeScript owns mechanism**: loops, sorting, counting, set/map joins, parsing, formatting, record assembly, and
  folder transaction checks live in `.ts` files or `@natlang/node/std` combinators.
- **Natural-language functions own judgment**: meaning, classification, extraction, intent synthesis, policy
  evaluation, conflict resolution, and open-ended planning live in typed `.nl` files and inline `nl<T>` lambdas.
- **No prose-TS recipes in root `.nl` files**: never write a root `.nl` that dictates step-by-step JavaScript code in
  backticks (`1. const d = await decide(...); 2. if (d.confidence >= 0.6) ...`). Either put that orchestration in a
  `.ts` helper/entry, OR give the root a real discovery/synthesis problem over a 10+ item mixed `.ts`/`.nl` library.

## 2. Hard Gates Every Task Must Pass

1. **Idiom-clean under `natlang check`**: zero `idiom-decision-readout`, zero `idiom-code-in-prose`, zero
   `idiom-fixed-cursor-iterate`, and zero `untrusted-instruction` findings.
2. **Real semantic oracle (beat the keyword baseline)**: every `oracle: verifier` task must include a genuine
   `keyword_baseline` that executes cleanly (`status === 'done'`) and **fails** the verifier, while the reference passes.
   No cue-word rules (`contains the word`, `by whole words`) in any `.nl` body.
3. **Executable artifact verifiers over tabular puzzles**: prefer the real-work verifiers under
   `training/semantic_fp_tasks/verifiers/`:
   - `code_test.mjs` — executes model-generated TypeScript/JS transforms or patches against hidden unit tests.
   - `policy_eval.mjs` — checks higher-order compiled rules (`compileRule` / `.nl` returning a function) over hidden
     cases + metamorphic paraphrase twins (`twins: [[idA, idB]]`).
   - `service_trace.mjs` — checks `ONCE_EFFECTS` idempotency, ordering, and prompt-injection resistance over service
     action logs.
   - `proper_score.mjs` — grades calibrated `readout: decision` probability distributions with Brier score, bounded
     log-loss, and asymmetric risk-floor escalation (`escalate` from `@natlang/node/std`).
   - `merge.mjs` — grades `semanticMerge` intent-preserving 3-way merges and explicit `unresolved[]` conflict reports.
4. **Custody hygiene**: `groups` holds ONLY real external source items (`<dataset>:<item_id>`). Never invent
   family-wide or theme-wide `invented:*` groups.
5. **Metamorphic paraphrase twins (O2)**: for semantic judgment tasks, include twin inputs with content-lemma Jaccard
   `< 0.25` and identical semantic outcome so cue-word shortcuts cannot pass both.

## 3. Target Task Archetypes (Prioritize These Over Mini-Desks)

| Archetype | Core Constructs | Preferred Verifier |
|---|---|---|
| Codebase discovery & composition | 12–30 mixed `.ts`/`.nl` helpers; root names none; `read_code` / `search_code` | `code_test.mjs` / exact |
| Higher-order policy compilation | `compileRule` / `.nl` returning `(item: T) => boolean`, `.partial()`, `withContext` | `policy_eval.mjs` (with `twins`) |
| Calibrated triage & risk escalation | `readout: decision`, `decide()`, `escalate(..., { floor, risk })`, `semanticPartition` | `proper_score.mjs` |
| Verified workspace & code repair | `proposeAndVerify`, `branchAndSelect`, `semanticRepair`, `protect:`, `edit_code` | `code_test.mjs` / `service_trace.mjs` |
| Untrusted input & relational contracts | `Untrusted<T>` with real prompt injections, `Is<T, P>`, `ensures:` (`check` + `predicate`) | `service_trace.mjs` / `policy_eval.mjs` |
| Structural recursion & semantic merge | `recursive: structural` / `measure:` with bottom-up synthesis; `semanticMerge` | `merge.mjs` / `code_test.mjs` |
| `@natlang/std` collection pipelines | `semanticFilter`, `semanticMap`, `semanticReduce` (`linear`/`tree`), `semanticMatch`, `semanticRank` | `code_test.mjs` / `proper_score.mjs` |
