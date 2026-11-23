# Technical & Vendor Diligence (`applications/diligence/`): Decomposition

Tier 3 flagship (WS6 §3 A10): evaluates untrusted vendor, library, or paper dossiers against a structured rubric using recursive bottom-up digestion with verbatim citation postconditions, asymmetric risk escalation, and debiased pairwise ranking.

## Policy vs Mechanism

- **Exact mechanism (TypeScript, `diligence.ts` and `digestSection/citationsMatch.ts`):**
  - Boundary tainting (`markDossierUntrusted`, `markSectionUntrusted`) marking every section body as `Untrusted<string>`.
  - Exact postcondition verifier (`digestSection/citationsMatch.ts`): guarantees every recursive digest preserves `sectionId`, `title`, and child ordering, and that every cited quote is non-empty and occurs verbatim in `section.text` or one of its descendant sections.
  - Concurrent orchestration (`semanticMap`), exact tallying of criterion verdicts (`dealbreakerCount`, `concernCount`, `strongCount`, `acceptableCount`), and debiased pairwise Copeland ranking (`semanticRank`).
- **Typed semantic policy (`.nl` leaves):**
  - `digestSection.nl` (`recursive: structural`, `ensures: citationsMatch(result, section)`): recursively digests a section tree bottom-up while grounding every claim in verbatim source citations.
  - `scoreCriterion.nl` (`readout: decision`, returns `CriterionVerdict`): calibrated rubric evaluation wrapped by `escalate(scoreCriterion, { to: 'teacher', floor: 0.82, risk: { dealbreaker: 0.2 } })`.

## Safety & Custody

- Every section body is `Untrusted<string>`, fencing vendor marketing copy and adversarial text from prompt injection.
- `citationsMatch.ts` is auto-protected by `ensures:` so hallucinated quotes are rejected at the tool boundary and repaired before returning.
- Asymmetric risk floor (`risk: { dealbreaker: 0.2 }`) routes suspected dealbreakers to `teacher` even when local argmax leans `'acceptable'` or `'concern'`.
- Fixture rows in `fixtures/cases.json` use `split_group: "flagship:diligence:<group>"`.
