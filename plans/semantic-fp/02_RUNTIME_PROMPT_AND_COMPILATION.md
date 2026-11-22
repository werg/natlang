# WS2: Runtime Semantics, Prompt Kernel, Factorized Decisions & Progressive Compilation

**Parent:** [Campaign Index](README.md)  
**Goal:** Cut the bloated 4,500-token universal system prompt down to a crisp semantic-FP kernel, give models scalable codebase discovery, extend `readout: decision` to small structured products across all four serving engines, and unify natlang's optimization machinery into one progressive compilation workflow.

---

## 1. Mechanism Inventory (Full Engine Parity)

| Mechanism | Components implementing / consuming it | Scope in WS2 |
|---|---|---|
| Model-facing system prompts & tool surface | `ts-host/src/native/prompt.ts`, `native/system-prompts.ts`, `native/agent.ts`, `native/tool-surface.ts`, `tests/neuralese/test_serving_prompt_rendering.py`, `training/neuralese/.../sft_sources.py` | Full parity across ts-host + Python SFT/RL prompt renderers |
| Codebase opening listing & `read_code` | `ts-host/src/native/agent.ts` (`callableDeclarations`, `largeCallableScope`), `native/runtime.ts`, `native/introspection.ts` | Progressive module summaries + `search_code` |
| Structured / factorized `readout: decision` | `ts-host/src/native/decision.ts`, `ts-host/src/model/decision-scoring.ts`, `ts-host/src/runtime/loader.ts`, **all 4 engines** (`docs/ENGINE_PARITY.md`): managed `llama-server` fork, in-page `wllama`, Python native engine (`serve/`), SGLang actor (`serve/sglang_actor.py`), plus `tests/neuralese/test_engine_parity.py` and `scripts/wllama_smoke.mjs` | Full parity across all 4 engines (tracked in `docs/ENGINE_PARITY.md`) |
| Confidence escalation (`escalate`) | `ts-host/src/runtime/escalate.ts` (new), `compiler/intrinsics.ts`, `@natlang/node`, `@natlang/browser` | Shared runtime helper + trace event `decision_escalate` |
| Progressive compilation (`profile`, `compile`, `RewriteGate`) | `ts-host/src/compiler/project.ts`, `compiler/rewrites.ts`, `calls/specializer.ts`, `calls/store-core.ts`, `cli/commands.ts`, `cli/calls.ts`, `dashboard/pages/` | Wire `RewriteGate` + add `natlang profile` and `natlang compile` |

---

## 2. Slim, Capability-Gated System Prompt Kernel

### 2.1 The Current Defect
`ts-host/src/native/prompt.ts` currently injects:
- `TOOLS_PROMPT` (~14.4 KB) + `GENERATION_GUIDANCE` (~0.9 KB) = **~15.3 KB** on every tool-loop call;
- `FUNCTION_TOOLS_PROMPT` (~1.6 KB) whenever a callable folder exists;
- `DIRECTORY_REDUCER_PROMPT` (~5.1 KB) on reducers.

This causes four direct harms:
1. **CPU latency & context tax:** ~4,000–6,000 tokens before a 3-line `.nl` function runs (forcing complex warm-prefix slot machinery just to survive startup).
2. **Benchmark scar-tissue overfitting:** Hyper-specific prose (`change-request.json`, `more: true` pagination, tree deep-copy rules, `"Refunds are allowed within 30 days"` — which LFM3-1.7B copied verbatim into 9/42 triage calls, prompting the hack `PROMPT_EXAMPLE_TEXTS` at `prompt.ts:121-124`).
3. **Anti-compositional bias:** `SEMANTIC_CALL_COMPLETION_PROMPT` (`prompt.ts:9`) pressures the model *not* to decompose work with child `nl` lambdas.
4. **Factual contradiction:** `APPROACH_PROMPT` (`prompt.ts:110`) tells the model ordinary function declarations do not persist across eval turns, contradicting `scope-compiler.ts:749-802`'s `persistentHelpers`.

### 2.2 Target Architecture (`promptProfile: 'kernel-v2'`)
Split the prompt into a **~400-token core kernel** plus **capability-gated micro-modules** (each ~80–150 tokens, registered in `system-prompts.ts` with stable IDs so soft-prompt banks and `program-header` rendering work cleanly):

1. **`kernel-core` (~350–450 tokens, always included):**
   - Identity: you are executing one typed natural-language function in a persistent TypeScript `eval` scope.
   - Tools: `eval(code, finish?)` for exact work, calling available functions/services, and composing inline `nl` lambdas; `return_result(status, value?, reason?)` to answer directly (`success` / `blocked` / `failed`).
   - Idiom: **Use existing functions in scope first; spawn focused typed inline `nl<T>\`...\`(arg)` for genuinely distinct semantic subproblems; keep exact logic, parsing, and aggregation in TypeScript; use `iterateOn` for open-ended convergence; finish computed values with `eval({ code: "return value;", finish: true })`.**
   - Scope rules: parameters are frozen; top-level `const`/`let` and pure helper functions persist across evals; cut-off previews `<<cut off: ...>>` mean the full value stays in the variable; finite loops only (`for...of`, counted `for`, array methods, `iterateOn`).
2. **Capability-gated micro-modules (included ONLY when the call actually uses them):**
   - `mod/codebase`: when `codebase` is non-empty (`read_code` / `search_code` to inspect modules & types).
   - `mod/edit-code`: ONLY when `mutable: true` / `codeEdits === 'allow'` (`edit_code`, `diff_code`).
   - `mod/folder`: ONLY when `subtype === 'directory-reducer'` or a parameter/return mentions `Folder`/`FileHandle` (concise `folder` handle API + `propose`/`accept`/`iterateOn`, stripped of `change-request.json` prose).
   - `mod/neuralese`: ONLY when any parameter, return, capture, or callee signature mentions `Neuralese<...>`.
   - `mod/decide`: ONLY when any callee has `readout: decision` (`decide(fn, ...args)` returns `{ value, probabilities, confidence }`).
   - `mod/transcript`: injected lazily only after turn 4 or compaction (`compact_history`, `transcript.search`).

### 2.3 Move Defensive Lessons into Actionable One-Line Runtime Diagnostics
Audit every defensive sentence removed from `TOOLS_PROMPT` and verify it is covered by a crisp runtime error message (already true for ~80% of them: `invalid-binding`, `decision-compare`, `undeclared-field`, `forbidden-loop`, cut-off copy check, placeholder rejection). Add missing one-line diagnostics where needed (e.g., mutating a frozen parameter → name the parameter and suggest `structuredClone(param)`).

### 2.4 Rollout & Scoring-Series Hygiene
- **Opt-out:** `promptProfile: 'legacy'` runtime option (and `NATLANG_PROMPT_PROFILE=legacy` env var) restores the prior `TOOLS_PROMPT` verbatim for A/B comparison.
- **Scoring series:** Per `AGENTS.md`, `kernel-v2` changes model-facing prompt text, so when adopted in a training pin, `grpo.py` moves the old step-0 aside and re-measures base `held-v3` on the `kernel-v2` series. Run a paired 2-sample `held-v3` comparison (`legacy` vs `kernel-v2`) on Pop before making `kernel-v2` the default for training; measure both pass rate and opening prefill/TTFT savings!

---

## 3. Progressive Codebase Discovery (Scaling to Real Multi-Module Codebases)

### 3.1 Current Bottleneck
`agent.ts:1378-1430` renders every callable item's full declaration into the opening message until `largeCallableScope` (`agent.ts:296-306`) abruptly truncates at >96 items. This wastes tokens on small calls and blinds the model on real codebases.

### 3.2 Target Design
1. **Two-tier opening declaration:**
   - **Direct callees (≤ 12 items):** Shown inline with full signature + 1-line JSDoc / `description`.
   - **Namespaces / larger libraries (> 12 items):** Shown as a compact **module directory** (`@module pricing (6 fns): quote, applyTier, validateCoupon, ... — use read_code("pricing") or search_code("...")`) with full signatures loaded on demand via `read_code`.
2. **Add `search_code(query)` helper in `eval` / tool surface** (offered when codebase has > 12 items): lexical + signature search over callable names, parameter/return types, `types.ts` aliases, service declarations, and JSDoc comments, returning matching signatures + paths.
3. **Update `read_code`:** `read_code("moduleName")` lists all exports with signatures and descriptions; `read_code("moduleName.fn")` shows full source/instructions; `read_code("TypeName")` shows the type alias with field doc comments (already partially implemented at `language.md:82`, standardize everywhere).
- **Opt-out:** `openingListing: 'full'` runtime option.
- **Targeted tests:** `ts-host/test/codebase-discovery.test.mjs` testing compact listing at 20+ helpers, `search_code` hits by return type & doc keyword, and on-demand `read_code`.

---

## 4. Factorized `readout: decision` & Calibrated Escalation (`escalate`)

### 4.1 Extending `readout: decision` Beyond Flat Primitive Unions
Today `native/decision.ts:19-37` and `runtime/loader.ts:299` restrict `readout: decision` to flat unions of ≥ 2 primitives. Many real routing/triage decisions are small finite products or tagged records:
```yaml
returns: '{ route: "mechanical" | "behavioral" | "security"; urgent: boolean; needsHuman: boolean }'
readout: decision
```
- **Mechanism (Factorized KV-Shared Scoring):**
  1. In `native/decision.ts`, add `isFiniteType(type, env)` supporting:
     - Primitive literal unions & `boolean` (cardinality ≥ 2).
     - Small records whose fields are all finite types, with total Cartesian cardinality ≤ 32 (configurable `maxDecisionStates`, default 32), OR sequential field-by-field factorized scoring when cardinality > 32 (score field 1's options → commit argmax or top-k beam → score field 2 conditioned on the shared prefix!).
     - Tagged unions of finite variants (`{ kind: "approve" } | { kind: "escalate", team: "sec" | "eng" }`).
  2. **Engine parity (`docs/ENGINE_PARITY.md`):**
     - `ts-host/src/model/decision-scoring.ts` builds the canonical JSON option strings (or sequential field-scoring requests) in **one shared place** for managed `llama-server`, in-page `wllama`, Python native engine, and SGLang actor!
     - Because all four engines *already* support shared-prefix option scoring (`cache_messages` / prefix checkpointing), small Cartesian products (≤ 32 options) require **zero C++/CUDA engine changes** — they score over the exact existing batch-score endpoint with prefix reuse!
     - For sequential factorized scoring (> 32 states, e.g., 3–6 finite fields), the client issues `k` prefix-cached scoring passes (1 per field, appending `"field": <chosen>,`), reusing the KV cache across every field on all 4 engines.
     - Add parity test cases to `tests/neuralese/test_engine_parity.py` and `scripts/wllama_smoke.mjs`.
  3. **Return shape of `decide(fn, ...args)`:** Preserves `{ value, probabilities, confidence, scored }` and adds `fieldProbabilities?: Record<string, { value, probability }[]>` for record decisions so callers can set per-field confidence floors (e.g., escalate only if `p(security) >= 0.25`!).

### 4.2 First-Class `escalate()` Combinator
Make the "90% on local 1.7B, 10% on frontier" pattern a one-line standard library primitive in `@natlang/node` and `@natlang/browser`:
```ts
import { escalate } from '@natlang/node';

const classify = escalate(classifyTicket, {
  floor: 0.85,                  // or per-value floors: { security: 0.20, default: 0.85 }
  fallback: 'frontier',         // model profile name in runtime `models`
  onUncertain: 'escalate',      // 'escalate' | 'return-with-flag'
});
```
- Runs `runtime.decide(fn, ...args)` on the fast/default model.
- Checks `confidence >= floor` (and any per-label risk floors, e.g., if `p('security') >= 0.20` escalate even when another label is argmax!).
- On low confidence or risk-floor trigger, automatically invokes `fn.on({ model: fallback })` and records a `decision_escalate` trace event (feeding `natlang profile` and `natlang tune`!).
- **Targeted tests:** `ts-host/test/factorized-decision.test.mjs` and `ts-host/test/escalate.test.mjs`.

---

## 5. Unified Progressive Compilation Ladder (`natlang profile` & `natlang compile`)

Unify natlang's existing optimization subsystems into one coherent developer loop:

```text
  .nl Specification (runnable immediately on any model)
         │
         ├──► natlang profile     (reads call store: calls, tokens, latency, JIT %,
         │                         confidence histogram, escalation %, hot pairs)
         │
         └──► natlang compile     (runs eligible compilation stages, records lock)
               ├── Stage 1: Trace JIT (`natlang specialize` -> guarded crisp TS cases)
               ├── Stage 2: Algebraic Fusion (`RewriteGate` from `natlang.json` -> map/read/combine rewrites)
               ├── Stage 3: Distillation / Tuning (`natlang tune` -> LoRA + soft states for LFM3-1.7B)
               └── Stage 4: Shadow Verification (`pluggable` shadow + live audit before promotion)
```

### Concrete Implementation Steps
1. **Wire `RewriteGate` into `natlang.json` (`compiler/project.ts:106, 205`):**
   - Add `"rewrites": { "mode": "off" | "measured", "measurements": ".natlang/rewrite-measurements.json", "disabled": [] }` to `natlang.json`.
   - Default remains `"off"` until a measurement artifact certifies `verdict: 'no-harm'` for `(model, dialect)` (`compiler/rewrites.ts:35-54`).
2. **Implement `natlang profile [DIR]` (`ts-host/src/cli/calls.ts` + dashboard Overview/Programs panel):**
   - Queries the machine call store (`calls/store-core.ts`) for every `.nl` and inline `nl` site in the project.
   - Prints a crisp table per function:
     `function | calls | p50 ms | tok/call | crisp JIT % | decide conf p10/p50 | escalated % | recommended action`
   - Actionable recommendations: *"High repeat structure → run `natlang specialize`"*, *"Finite union without readout → add `readout: decision`"*, *"High escalation rate (34%) → run `natlang tune` on 120 collected traces"*.
3. **Implement `natlang compile [DIR] [--specialize] [--tune] [--verify]`:**
   - Orchestrates specialization + optional tuning + shadow audit into `.natlang/build/natlang.lock.json` with atomic rollback (`natlang compilations` / `natlang tune` existing rollback primitives).
- **Targeted tests:** `ts-host/test/profile-compile.test.mjs` (synthetic call store fixtures -> profile metrics -> RewriteGate toggle -> lockfile provenance).

---

## 6. Acceptance Gates for WS2

- `kernel-v2` prompt is **≤ 600 tokens** (down from ~4,500) on a plain function call, with `legacy` opt-out bit-identical to previous prompt.
- Paired `held-v3` run on Pop comparing `legacy` vs `kernel-v2` shows no quality regression (p ≥ 0.05 or positive gain) and measurable TTFT/prefill reduction.
- Factorized `readout: decision` passes `tests/neuralese/test_engine_parity.py` and `scripts/wllama_smoke.mjs` across all 4 engines.
- `natlang profile` produces accurate per-site cost/JIT/escalation reports on `examples/specialization` and `applications/helpdesk`.
