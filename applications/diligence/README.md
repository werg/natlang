# Technical & Vendor Diligence (`@natlang/app-diligence`)

Grounded rubric evaluation and debiased ranking over untrusted vendor, library, and technical paper dossiers.

## Architecture

- **`types.ts`**: `Section` (`text: Untrusted<string>`), `Dossier`, `SectionDigest`, `CriterionVerdict`, `DossierScore`, `DiligenceReport`.
- **`digestSection.nl`** (`recursive: structural`) + **`digestSection/citationsMatch.ts`**: bottom-up recursive section digestion held to an exact verbatim citation postcondition.
- **`scoreCriterion.nl`** (`readout: decision`): scores a grounded digest against each rubric criterion.
- **`diligence.ts`**: digests untrusted dossiers concurrently, scores every rubric criterion via `escalate(..., { to: 'teacher', floor: 0.82, risk: { dealbreaker: 0.2 } })`, and ranks candidates via debiased `semanticRank`.

## Verification

```bash
node ts-host/bin/natlang.mjs check applications/diligence
node --test ts-host/test/flagships-tier3.test.mjs
```
