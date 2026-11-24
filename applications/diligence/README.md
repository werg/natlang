# Technical & Vendor Diligence (`@natlang/diligence`)

Grounded rubric evaluation, weighted scoring, Neuralese section compaction, and debiased ranking over untrusted vendor, library, and technical paper dossiers.

## Architecture

- **`types.ts`**: `Section` (`text: Untrusted<string>`, optional `viewSlot`), `Dossier`, `SectionDigest`, `CriterionVerdict`, `DossierScore` (with `weightedScore` and `recommendation`), `DiligenceReport`.
- **`digestSection.nl`** (`recursive: structural`) + **`digestSection/citationsMatch.ts`**: bottom-up recursive section digestion held to an exact verbatim citation postcondition.
- **`scoreCriterion.nl`** (`readout: decision`): scores a grounded digest against each rubric criterion.
- **`diligence.ts`**: fences every dossier at entry via `markDossierUntrusted`, digests dossiers concurrently (with optional `carry: 'neuralese'` / `neuraleseView` compaction), scores every rubric criterion via `escalate(..., { to: 'teacher', floor: 0.82, risk: { dealbreaker: 0.2 } })`, computes weighted scores and recommendations using `RubricCriterion.weight`, and ranks candidates via debiased `semanticRank`.
- **`main.ts`**: runnable CLI target (`natlang run applications/diligence -- input.json`).
- **`tune.examples.json`** & **`fixtures/cases.json`**: structured tuning examples and split-labeled benchmark fixtures.

## Verification

```bash
node ts-host/bin/natlang.mjs check applications/diligence
node --test ts-host/test/flagships-tier3.test.mjs
```
