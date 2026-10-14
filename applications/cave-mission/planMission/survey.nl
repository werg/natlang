---
description: Reduce nested survey collections into distinct frame, passage and equipment assertions.
kind: directory-reducer
args:
  request: MissionRequest
returns: SurveyEvidence
---
1. Read manifest.json with folder.readText and parse SourceManifest. Require its caseId to equal
   request.caseId. The manifest lists the finite source inventory and its visible identity conventions.
2. Visit the listed documents grouped by their real parent directories. Read exact text, preserve
   manifest id/path/roleHint, and extract every visibly labeled clause into SourceClause with a
   verbatim Witness. Directory and role hints guide reading without overriding what a clause says.
3. Record per-document clause IDs in coverage. If duplicate IDs disagree, retain the witnessed
   conflict in unresolved instead of overwriting an assertion. Keep negation and exception wording.
4. Call readFrames(clauses) and readEquipment(clauses) and consume their actual results. Then call
   readPassages(clauses, frames) with that returned FrameSurvey. These children own coordinate,
   directed-passage and carried-state/sensor meanings; passage endpoints use the actual frame keys.
5. Preserve a clause in several relevant branches when it truly couples them. For example, a chimney
   assertion can constrain both direction and rope requirements. Quote the original clause in both.
6. Combine the returned results into SurveyEvidence. Include the unchanged source documents and
   manifest so later witnesses can be checked exactly without consulting an expected-answer file.
7. Keep global and local landmark labels distinct at this stage. Binding a local landing to a global
   side belongs to reconciliation; source reading must not quietly choose the missing frame convention.
8. Return all evidence and unresolved source questions. The energy allowance and costs are task
   facts; they must not be converted into a budget for reasoning or a limit on model calls.
