# Saga Forensics

A natural-language application that investigates an immutable distributed transaction snapshot and returns a contingent repair policy as data. It does not contact a bank or warehouse.

The entry is the directory reducer `investigate(folder, request): ForensicReport`. `folder` is the task's evidence root containing `manifest.json`; `request` has type `InvestigationRequest` from `types.ts`. The source project and task evidence are separate snapshots. The expected answer and review receipts are never placed in the model-visible folder.

## First implemented world

The pooled-rental world combines warehouse epoch identity, bank tenant identity, semantic retries, rejected incompatible reuse, clock offsets, partial custody, shared stock, a protected deposit and a refund with two supported outcomes. It requires a policy whose observation branch changes the necessary write while both branches end in the same justified stock and cash state.

The authored program has 23 named NL functions. Two `iterateOn` reducers shrink finite candidate/obligation frontiers. Four scoped NL literal sites judge payload meaning, evidence strength, query-answer scope and compensation scope. These judgments feed typed state used by the caller. See [DECOMPOSITION.md](DECOMPOSITION.md) for ownership and exact seams.

## Status

All 23 functions load through the real source loader with their callable-folder edges and inherited typed aliases. Type declarations compile. These checks establish source validity only. Independent source/oracle review, exact result-verifier replay and actual live execution remain pending. No training admission or measured model-facing adoption is claimed.

The task manifest is `training/semantic_applications/astra/saga-forensics/project.json` in the repository. The corpus packager preserves the authored project rather than rendering a generic stage chain. Future task variants sharing this evidence world use the same split group.
