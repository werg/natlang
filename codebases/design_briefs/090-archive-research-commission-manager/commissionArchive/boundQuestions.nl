---
description: Derive constrained historical questions with evidence paths and inference ceilings.
args: { request: CommissionRequest; checkpoint: ResearchCheckpoint }
returns: { questionIds:string[]; questions:string[]; inferenceLimits:string[]; stoppingRule:string; }
---
Separate factual lookup from interpretive synthesis. Name what evidence could answer each question and what it cannot establish. Retain exclusions and finite stopping rule. Reuse identifiers for unchanged checkpoint questions and replace identifiers for changed scope.
