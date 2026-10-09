# Refine data: decomposition note

Status: draft for owner review (plans/OWNER_REVIEW.md). This package is a teacher-side data generator, not a
user-serving app, so this is a short note. Nothing is restructured until the owner has reviewed it.

The package is three natural-language stages (`exemplify.nl`, `nearMiss.nl`, `verifyNearMiss.nl`), `index.ts:9-12`
(exports only), `types.ts` and `authoring.json` (the registry of guard sentences and the type slots that replace them).
The crisp parts that turn stage outputs into rows live outside the package (`ts-host/scripts/refine-data/`,
`scripts/refine_data/`, see plans/REFINEMENT_DATA.md).

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| Satisfying values for a predicate | fn | `exemplify.nl` | Generation. Contract `(predicate, base, slot, count) -> Examples`. |
| Minimal edit that breaks the predicate | fn | `nearMiss.nl` | Generation. One detail changes; the instruction states it positively. |
| Independent second pass over the pair | fn | `verifyNearMiss.nl` | Verification by a separate call, as the family requires. |
| Rows from stage outputs (labels, pairs, held flags) | crisp | `scripts/refine_data/`, `ts-host/scripts/refine-data/lib.mjs` | Data assembly, exact. |
| Exact labels for predicates that a crisp check decides | crisp | `scripts/refine_data/exact.py` | Exact verifier; labels come from the check, not a model. |
| Guard-to-type registry | data | `authoring.json` | Maps each guard sentence to a refinement slot; omissions are reported. |
| Admission of produced corpora | host | `training/neuralese_corpora.json` | Explicit; this package never grants admission. |

Points that follow the natlang-native rules:

- **The stages stay three functions.** Each is one task for a small model, and `verifyNearMiss` has to be a separate
  call to be independent of the writer.
- **`verifyNearMiss` has crisp support where the predicate is crisp.** When the refinement predicate has a crisp
  checker, the verdicts `original_holds` and `edited_holds` come from the checker and the model only supplies
  `minimal` and `reason`. This is already how the exact family labels rows.
- **New source for `authoring.json`.** The refinement-candidate tables in the new app decompositions (helpdesk,
  terminal, media, notebook, evidence, publisher, ide, specializer, program-improver) name guard sentences and the
  `Is<T, "...">` slots that replace them. Adding those pairs to `authoring.json` makes them `refine-author` rows,
  with the registry reporting any whose `find` text is missing. This follows the owner review of each app; nothing
  is registered before then.
- **Model-facing text:** the three `.nl` files are already positively phrased. No change is proposed.
