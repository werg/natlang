# Technical & Vendor Diligence (`applications/diligence/`): Decomposition

Tier 3 flagship (WS6 §3 A10): evaluates untrusted vendor, library, or paper dossiers against a structured rubric using recursive bottom-up digestion with verbatim citation postconditions, asymmetric risk escalation, and debiased pairwise ranking.

## Policy vs Mechanism

- **Exact mechanism (TypeScript, `diligence.ts` and `digestSection/citationsMatch.ts`):**
  - Boundary tainting (`markDossierUntrusted`, `markSectionUntrusted`) marking every section body as `Untrusted<string>`.
  - Exact postcondition verifier (`digestSection/citationsMatch.ts`): guarantees every recursive digest preserves `sectionId`, `title`, and child ordering, and that every cited quote is non-empty and occurs verbatim in the text of the section its `sectionId` names.
  - Concurrent orchestration (`semanticMap`), exact tallying of criterion verdicts (`dealbreakerCount`, `concernCount`, `strongCount`, `acceptableCount`), and debiased pairwise Copeland ranking (`semanticRank`).
- **Typed semantic policy (`.nl` leaves):**
  - `digestSection.nl` (`recursive: structural`, `ensures: citationsMatch(result, section)`): recursively digests a section tree bottom-up while grounding every claim in verbatim source citations.
  - `scoreCriterion.nl` (`readout: decision`, returns `CriterionVerdict`): calibrated rubric evaluation wrapped by `escalate(scoreCriterion, { to: 'teacher', floor: 0.82, risk: { dealbreaker: 0.2 } })`.

## Status and limits

Not built yet from plan 06 A10: `Neuralese<T>` views of large sections, `natlang tune` on past verdicts, and rubric
weights (`RubricCriterion.weight` is carried but unused; the final order is `semanticRank`'s judgment). No fixture pack:
the earlier `fixtures/cases.json` named candidates and an expected winner without any dossier text, so nothing could
run it; it was removed.

## Safety & Custody

- Every section body is `Untrusted<string>`, fencing vendor marketing copy and adversarial text from prompt injection.
- `citationsMatch.ts` is auto-protected by `ensures:` so hallucinated quotes are rejected at the tool boundary and repaired before returning.
- Asymmetric risk floor (`risk: { dealbreaker: 0.2 }`) routes suspected dealbreakers to `teacher` even when local argmax leans `'acceptable'` or `'concern'`.
