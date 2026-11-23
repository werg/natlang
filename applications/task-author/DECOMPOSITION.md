# Task Author (`applications/task-author/`): Decomposition

Tier 0 flagship tool (WS3 T3, WS6 §2): authors and qualifies Semantic-FP RL/SFT training tasks from seeds with
exact or verifier-backed oracles, ensuring every admitted task is genuinely semantic and defeats trivial keyword
baselines.

## Policy vs Mechanism
- **Exact mechanism (TypeScript, `author.ts`):**
  - Enforces deterministic structural invariants: required files (`natlang.json`, `types.ts`, entry module),
    construct tag presence, valid verifier/oracle specification, held-v4 reservation exclusion (`split_group`),
    and a deterministic keyword-baseline attack check.
  - Drives drafting through `semanticRepair` (passing exact structural + baseline failures back as `problems`)
    and aggregates qualification results.
- **Typed semantic policy (`.nl` functions):**
  - `draftTask.nl`: synthesizes a concrete Semantic-FP task bundle (`files`, `prompt`, `oracle`, `constructs`)
    from a domain seed, target construct family, and any prior rejection `problems`.
  - `isSemantic.nl` (`readout: decision`): judges whether the authored task genuinely requires semantic judgment
    rather than keyword matching or mechanical parsing.
  - `reviewOracle.nl` (`readout: decision`): judges whether the task's verifier/exact oracle faithfully grades the
    intended semantic outcome without leaking the answer or admitting trivial shortcuts.
