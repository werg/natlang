# P07 — Natlang package manager

Status: proposed implementation. [Shared capabilities](README.md).

## Natlang prerequisites

C0 expresses package selection and migration reasoning. C3 supplies codebase construction and validation independent of a physical directory. C1 supplies selected filesystem/archive/network bindings. C7 remains the single boundary type language. Package resolution is exact library code, not an eval-time implicit installer or new runtime primitive.

## Programme and typed boundary

`resolve.nl(request, candidates, policy) -> ResolutionReport`, `install.nl(lock, target) -> InstallReport` and `upgrade.nl(snapshot, constraints) -> UpgradePlan`. Natlang chooses relevant candidates and interprets compatibility; crisp helpers solve constraints, verify hashes and materialise the selected tree.

Ordinary records describe versions, dependencies, exported function signatures, effects, required engines/environment contracts, source hashes and scenario evidence. A lock pins exact transitive content and engine requirements. Native archive/file/index objects stay in the crisp environment.

## Crisp environment

Expose small helpers for reading a local index, checking sources, solving constraints, fetching by identity and atomically publishing an installed tree. Begin with local directories/archives and current `uses` resolution. No public registry or network is necessary for the first slice.

An install does not execute arbitrary lifecycle scripts or silently change engine authority. If building an installed package is requested, route that explicit operation through the selected execution environment. Shared-host packages requiring native access must declare that environment dependency; they cannot be described as equally portable to a value-only sandbox.

## Reduction and loading behavior

Resolve and validate before starting the consuming programme. A running lambda retains its immutable lexical codebase. Upgrading creates a new source/lock revision used by a subsequent run; no hot-code mutation or new dynamic call semantics.

Dependency graph traversal uses crisp worklists or bounded natlang combinators. Installation state belongs to the package application. Recovery after interrupted publication preserves the previous installed tree and exposes the incomplete attempt; it is not a general runtime transaction feature.

## Delivery and checks

1. Package two existing helper libraries and resolve/install from an offline index.
2. Reconstruct the same checked codebase from the lock in another directory/in-memory bundle. Gate: identical exported definitions and compatible engine closure.
3. Add incompatible type/effect/engine changes and a proposed migration. Gate: exact incompatibilities are found and semantic compatibility claims cite scenario evidence.
4. Add remote registry/publication only if desired, preserving the same content identity contract.

Test conflicting constraints, duplicate exports, dependency cycles, missing engine, corrupted content, path traversal and interrupted install. A successful structural check is not proof of behavioral compatibility.

## Trace and teacher

Trace candidate selection, solver results, exact locked content and changed source revisions. Teacher targets cover choosing candidates, interpreting incompatibility and proposing constrained migrations; exact solving supplies the oracle. P20 can later apply accepted upgrade plans. No target running an already installed programme needs to carry this resolver or registry implementation.
