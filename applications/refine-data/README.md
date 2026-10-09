# Refine Data

Teacher-side stages that build the `refine-judge` training family (`plans/REFINEMENT_DATA.md`).

- `exemplify.nl`: satisfying values for a predicate (seed values for the harvested predicates).
- `nearMiss.nl`: a satisfying value, edited minimally into a violating one.
- `verifyNearMiss.nl`: the independent second pass over the pair.
- `index.ts`: exports the three functions.

Run them with `ts-host/scripts/refine-data/run-stage.mjs` in the teacher window; the commands are in
`plans/REFINEMENT_DATA.md`. Tests: `ts-host/test/refine-data.test.mjs` (scripted model).
