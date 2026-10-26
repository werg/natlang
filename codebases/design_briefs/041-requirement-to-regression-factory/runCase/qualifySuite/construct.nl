---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftSuiteQualification
---
Construct a SuiteQualification draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Run independent seeded defects and mutation tests to evaluate whether the suite detects meaningful violations. Remove redundant tests only when coverage remains, preserve regression cases for repaired bugs, and report uncovered requirements explicitly.
