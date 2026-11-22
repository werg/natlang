# WS1: Language — types, higher-order semantic functions, modules, recursion, contracts, malleable code

**Parent:** [Campaign index](README.md). **v2** (2026-11-23): claims re-verified against main 0f87d11a4. The
corrections are marked *(v2 fact)*. Sizes are agent-days (1 = one agent session of ~6-8 h).

**Goal.** Make the language able to express what semantic FP needs, without workarounds: generic semantic combinators
over hard and soft (`Neuralese<T>`) values, functions that take and return semantic lambdas with checked signatures,
cross-package semantic libraries, structural recursion in `.nl`, contracts that relate a result to its inputs, and
programs whose code the model may change, through a transaction that exact checks guard.

## 1. Mechanism inventory (every component the WS1 items touch)

| Mechanism | Components implementing or consuming it | In scope |
|---|---|---|
| Type grammar, validation, schema | `native/types.ts`, `native/values.ts`, `native/agent.ts` (`schemaOf`, `formatType`), `native/representation.ts`, `compiler/targets.ts`, `compiler/intrinsics.ts`, `compiler/eval-check.ts` | all |
| Inline `nl` planning and lowering | `compiler/inline.ts`, `compiler/holes.ts`, `compiler/lower.ts`, `runtime/lowered.ts`, `runtime/callable.ts`, `runtime/surface.ts` | all (Node, browser, compiled projects, eval) |
| `.nl` loading, `uses:`, imports | `runtime/loader.ts`, `runtime/modules.ts`, `runtime/contexts.ts`, `runtime/virtual-project.ts` (`loadVirtualCallables`), `compiler/project.ts` (`.d.nl.ts` emit) | all |
| Recursion and termination | `runtime/kernel.ts:526-545`, `runtime/context.ts:13` (`MAX_AD_HOC_NL_DEPTH`), `:223-264` (`guard`/`descend`), `runtime/hooks.ts:22`, `compiler/policy.ts`, `runtime/iterate.ts` | all |
| Refinement and trust | `native/refinement.ts:79-97`, `native/untrusted.ts`, `compiler/refined-slots.ts`, `compiler/inline.ts`, `runtime/kernel.ts` | all |
| Code edits in a call | `native/agent.ts:1276` (offer), `native/runtime.ts:1438-1445` (`codeEditsDenied`, `codeEditable`), `:2089-2114` (edit path), `adaptation/program-view.ts` (`ProgramView` revisions), `runtime/loader.ts:150` (`withhold:`), `improvement/adoption.ts` (adopt/rollback) | all |
| Model-facing text that describes these | `native/prompt.ts` (`TOOLS_PROMPT`: "functions may not call themselves recursively"; `FUNCTION_TOOLS_PROMPT`), `native/runtime.ts:270` (`read_code` builtin docs), `native/system-prompts.ts` piece registry | each item updates its sentence in the same commit and records it in `plans/MODEL_FACING_CHANGES.md` |
| Python renderers of the same prompt | `tests/neuralese/test_serving_prompt_rendering.py`, `scripts/neuralese_data/sft_sources.py` (renders recorded prompts; they carry the prompt as data) | prompt text changes need no Python edit because renderers replay the recorded system message. The test pins that and must stay green. |
| Serving engines | Type and loader changes reach engines only as rendered prompt text and tool JSON schemas (`agent.ts` `schemaOf`). Tuple `prefixItems` (L1) is a schema change: guided decoding on every engine must accept it. | L1 checks grammar-constrained decoding of `prefixItems` on llama-server/wllama (GBNF from JSON schema), Python native (`serve/native_guidance.py`) and SGLang (xgrammar). A gap goes into `docs/ENGINE_PARITY.md` with its blocker. |
| Training consumers | `scripts/rl_tasks/sources/authored.py`, `authored_checks.py`, `verify.mjs`, `structure_smoke.mjs` | new frontmatter keys (`ensures:`, `measure:`, `recursive:`, `generic:` forms, `protect:`) parse in authored-task checks |
| Docs | `spec/SPEC.md`, `spec/ERRORS.md`, `spec/ext/refinements.md`, `spec/ext/neuralese.md`, `skills/natlang-authoring/references/{language,refinements}.md` | same push as each feature (AGENTS.md "Teach in errors": every new error has recovery advice in the skill) |

## 2. Done (Wave 0, 2026-11-22, `ts-host/test/semantic-fp-fixes.test.mjs`)

- `Is<T, P>`, nested `Is`, `Untrusted<Is<T, P>>` and record intersections in inline `nl` (`compiler/targets.ts:166-202`).
- `.in(context)` typed on `NatlangCallableMethods`, `InlineNatlangFunction`, `NatlangUntypedFunction`.
- `Folder.apply/propose/iterateOn` accept crisp reducers; `moveTo` takes `string | Folder | FileHandle`.
- APPROACH_PROMPT's helper-persistence sentence matches `scope-compiler.ts`. *(v2 fact: APPROACH_PROMPT is not sent
  at runtime. It is used only by `teacher/cli.ts --approach-guide` and the piece registry, so this fix changed
  teacher prompts, not executor prompts.)*
- `crisp.mjs`'s `while` replaced.

A separate agent is reviewing this Wave-0 code. Its findings land as fixes, not as a plan change.

## 3. Work items

### L1. Tuples, `Record<LiteralUnion, V>`, `Partial/Required/Pick/Omit` (2 days, no deps)
- Defect: `compiler/targets.ts` flattens `[A, B]` to `(A | B)[]`; `native/types.ts` has no tuple node, so
  `natlang:neuralese`'s `zip` (`Promise<N<[A, B]>>`, `compiler/project.ts:212`) has no faithful type. `Record<K, V>`
  rejects literal-union keys.
- Build: `{ kind: 'tuple', elements }` through parser, `formatType`, `fitsType`, `values.ts` `coerce`, `schemaOf`
  (`prefixItems`, `minItems`/`maxItems`), `targets.ts`; literal-key records desugar in `TypeEnv.resolve`; the four
  utility types resolve over record types.
- Tests: `ts-host/test/types-extended.test.mjs` (parse, fit, coerce, schema, `zip` round trip). Engine check as in
  the inventory.

### L2. Static `iteration-unbounded`; `untrusted-instruction` in eval (1 day, no deps)
- `compiler/policy.ts`: an `.iterateOn(...).until(tsPredicate)` chain without `withMeasure`/`withLimit` is reported
  at `natlang check` (runtime check stays).
- `compiler/eval-check.ts`: interpolating a known `Untrusted<T>` binding into eval-snippet `nl` text is
  `untrusted-instruction`. Today only compiled sources get it (`spec/ext/refinements.md:92-93`).
- Tests: `ts-host/test/policy.test.mjs`, `untrusted.test.mjs`.

### L3. Malleable code through a transaction (3 days, no deps). Replaces v1's "edit_code opt-in"
*(v2 fact)* `edit_code` is not offered to every function. It is offered when the call has its own codebase (companion
folder or `uses:` attachments) and edits are not denied (`native/runtime.ts:1444`). Evaluation, self-play, improvement
and teacher runs already pass `codeEdits: 'deny'`. Frontmatter `withhold: [edit_code, diff_code]` exists. Edits do
not touch disk: they become revisions in the run's `ProgramView` (`adaptation/program-view.ts`). The stale comment at
`native/runtime.ts:1990` ("offered to every call") is wrong and is fixed here.

**Decision: keep `edit_code` on by default and make its edits transactional.** Malleability is the thesis: a program
that can repair its own helpers is the product. The real defect is not that edits exist. It is that (a) a stuck model
can edit the thing that judges it, and (b) a good edit dies with the run, because nothing proposes it for adoption.
v1's opt-in would remove the capability from every program to stop (a), and would do nothing about (b).

Build:
1. **Protected items.** A verifier cannot be a helper. Anything that judges the call is not editable in it: items
   named by `ensures:` checks (L7), items listed in frontmatter `protect: [path, ...]`, `types.ts`, and every
   service. Editing one is rejected with a one-sentence teaching error, e.g. "`checks/citations.ts` decides whether
   this call succeeded, so it cannot be edited here; fix the code that produces the value it rejects." (AGENTS.md:
   teach in errors). Skill item 5's existing rule ("make the outside world a service") stays the primary guidance.
2. **Run-scoped revisions are proposals.** Each successful call that committed revisions records them in its call
   record (`code_revisions: [{ source, originalHash, patchedHash, patch }]`; trace event already carries the hashes,
   `native/runtime.ts:2114`).
3. **Adoption is a Folder transaction.** `natlang improve adopt-call CALL_ID [--into DIR]` builds a `Folder`
   proposal from the call's revisions. It runs `natlang check` and `natlang test` on the proposal snapshot and
   `accept`s only if both pass and no protected item changed. Otherwise it reports the failing check. It reuses
   `improvement/adoption.ts` (adopt/rollback). The dashboard's trace page shows "adopt this fix". This is the same
   transaction `natlang improve DIR` (WS8 N4) uses: one implementation.
4. Opt-outs stay as they are (`codeEdits: 'deny'`, `withhold:`). The default stays `allow`.
- Model-facing: `FUNCTION_TOOLS_PROMPT`'s "Use edit_code to fix it when that tool is offered" is unchanged; the
  protected-item error is new text, recorded in MODEL_FACING_CHANGES.
- Tests: `ts-host/test/code-edits.test.mjs` (protected rejection, revision record, adopt-call accept and reject,
  rollback byte-identical); `conformance/programs/17-reopen-refine.yaml` stays green unchanged.

### L4. Higher-order semantic functions with checked signatures (3 days, no deps)
- *(v2 fact)* `nl-not-called` (`compiler/inline.ts:370-376`) rejects an arrow whose body is a bare `nl` tag only when
  that arrow is passed as a call argument (`items.map(item => nl\`…\`)`). A factory in a variable declaration is
  already accepted. `:381-389` rejects `await nl\`…\`` unless saved to a callable-typed variable.
  `native/values.ts:200-202` accepts any `typeof raw === 'function'` for a lambda type.
- Build:
  1. Allow an uncalled `nl` in return position, in a property typed as a function, and as an argument to a parameter
     whose type is callable. Keep the error for an uncalled `nl` used as a value nobody can call.
  2. Tag every natlang callable (inline, named `.nl`, callable-folder TS export) with its runtime `Type` under a
     non-enumerable `NATLANG_SIGNATURE`. In `values.ts`, check `fitsType(signature, wanted)`
     (contravariant params, covariant result). Wrap an untagged JS function in a boundary proxy that coerces its
     arguments and result at each call.
  3. `fn.partial(bound)` on natlang callables, preserving `.iterateOn`, `.in`, `.on` and the narrowed signature.
  4. A `.nl` with a lambda `returns:` is told to finish with `eval({ code: "return nl<…>`…`;", finish: true })`.
     That sentence goes in its call notes, not the system prompt.
- Tests: `ts-host/test/higher-order.test.mjs`.

### L5. Generics, polymorphic over hard and soft values (4 days, after L4)
- *(v2 fact)* `compiler/targets.ts:127` rejects type parameters. `native/representation.ts:26-43` allows exactly one
  `generic:` parameter constrained to `T | Neuralese<T>`.
- Build: frontmatter `generic: { T: unknown, K: string, S: 'X | Neuralese<X>' }`. Call sites are monomorphised from
  the TS checker's resolved signature, and the call record carries the instantiation. Generic TS wrappers around
  inline `nl` get `typeOf<T>()` injected at monomorphic call sites. The model always sees concrete types. The existing
  `R: T | Neuralese<T>` form becomes one case of the general rule, with the same diagnostics.
- This is what lets `@natlang/std` combinators (WS3) carry `Neuralese<S>` state (WS8 N3) with one signature.
- Opt-out `--no-generic-nl` during rollout; removed once WS3's combinators depend on it (owner: no undecided modes).
- Tests: `ts-host/test/generic-nl.test.mjs` (instantiation, constraint violation, nested generics, `S | Neuralese<S>`
  instantiated both ways, call-record provenance).

### L6. Modules: aliased and cross-package `uses:`, shared imports, compact declarations (3 days, no deps)
- *(v2 fact)* `uses:` takes package-root-relative paths (`harness/cut`, `runtime/loader.ts:290-292`), attaches each
  under its basename (collisions refused, `:346-348`), and is package-local. The import restriction is in
  `runtime/modules.ts:264-271` ("only items in this callable folder can be imported").
- Build: map form `uses: { git: "@natlang/lib-git/inspect", rubric: "./shared/rubric" }` (the list form stays the same
  thing with basename keys). Relative and package specifiers resolve through `natlang.json` package exports, which is
  how `@natlang/std` ships (WS3). The loader builds the whole call graph and refuses a cycle with its path unless L7's
  structural recursion admits it. Callable-folder `.ts` may import package-relative modules and natlang package
  exports under the same loop policy. `.d.nl.ts` emits `import type` references instead of one inlined intersection
  (`compiler/project.ts:160-193`).
- Tests: `ts-host/test/uses-modules.test.mjs`.

### L7. Structural recursion in `.nl`; `ensures:` contracts (4 days, no deps; two agents in parallel)
- **Recursion.** *(v2 fact)* `runtime/kernel.ts:545` refuses any re-entry of a call identity. TS already has a
  structural-descent guard (`runtime/context.ts:223-264`). Build: frontmatter `recursive: structural` or
  `measure: <expr>`. Re-entry is admitted when an argument strictly descends or the measure strictly decreases, and the
  same guard is reused. The function's own name is bound in its eval scope. `TOOLS_PROMPT`'s "functions may not call
  themselves recursively" becomes true-to-semantics: "a function may call itself only on a smaller argument". That is
  a model-facing fact fix recorded in MODEL_FACING_CHANGES. `MAX_AD_HOC_NL_DEPTH` stays 5 for ad hoc inline calls
  (`runtime/context.ts:13`). v1 proposed a size-shrink heuristic, which is a premature limit dressed as a feature.
  This item drops it. Tests: `ts-host/test/nl-recursion.test.mjs`.
- **`ensures:`.** *(v2 fact)* the refinement judge sees only `value` and `predicate` (`native/refinement.ts:79-97`).
  Build: frontmatter `ensures: [{ check: "<TS expr over result and args>", message }, { predicate: "<NL>", against:
  [args] }]`. Crisp checks run first (read-only scope). Semantic predicates run as a `readout: decision` judge whose
  user message carries `{ value, context }` with `Untrusted<T>` fencing kept. Failures re-enter the existing repair
  budget. Items named by a `check:` are protected (L3). `refine(value, predicate, { context })` gets the same
  context path. Tests: `ts-host/test/relational-refinements.test.mjs`.

### L8. `Neuralese<T>` is first class in the surface authors touch (2 days, after L5)
*(v2 fact)* `Neuralese<T, D>` is already a first-class, specified type (`spec/ext/neuralese.md` 0.5-draft: opacity
rules, conversions, dialects, `view`, `read`, combinators with laws). The gap is reach, not definition: 0 `.nl` files
use it, no std combinator accepts it, and the generic form is a special case. Build: L5 generalises the generic form.
`skills/natlang-authoring/references/language.md` gets a Neuralese section. `natlang check` hints at a
`Neuralese<string>` `view` when a callable passes the same large string to more than N child calls in one eval. The
hint is telemetry-derived and no limit is imposed. Tests in `generic-nl.test.mjs` and `neuralese` conformance.

## 4. Acceptance for WS1
- `npm test` and `npm run test:browser` in `ts-host` pass, including each item's targeted suite. Browser and Node give
  identical diagnostics.
- `spec/SPEC.md`, `spec/ERRORS.md`, the extensions and the skill references are updated in the same push as each item.
  Every new error code has a one-sentence fix in the skill's error table.
- `conformance/programs/` and existing application tests stay green.
