# P11 — Collaborative executable wiki

Status: local collaborative wiki slice in `codebases/wiki/` and
`applications/wiki.mjs`. [Shared capabilities](README.md).

Natlang composes the meaning of concurrent stable-block edits. The host pins
the merge model/source/seed profile, validates update IDs and coverage, checks
cell source shape, and retains unresolved alternatives. A page cell runs as a
child source graph with explicit input and no inherited wiki host authority.
Its result is pinned to page and source revisions; a late result after a merge
is marked stale. Tests cover concurrent edits, quickjs and natlang cells,
profile mismatch, invalid merged code and late results.

This is a local simulation. The user-requested notional CRDT has no crisp
convergence law. Cross-replica repeatability with a live pinned model,
transport/reconnection, actual browser inference and UI rendering remain
unproven. The two-client test uses a scripted model, so it checks the product
boundary rather than semantic merge quality.

## Natlang prerequisites

C2 supplies P02's shared merge execution profile. C3 validates and invokes page/cell source as data. C5 receives edits and execution results. C1 supplies the selected browser/server environments; C4 links revisions and reductions. C6 is optional for independent cells. No mutable lexical codebase, ambient interrupt or new collaborative type primitive is required.

## Programme and typed boundary

`handle.nl(state, event) -> WikiState`, `merge_page.nl`, `render_page.nl`, `run_cell.nl` and `review_revision.nl`. A page consists of stable prose/block/cell IDs and revisions. Cell records declare source, explicit inputs/dependencies and host requirements. Natlang performs semantic merges and page behavior; crisp helpers parse document boundaries, validate source, apply checked patches and render views.

Native DOM/editor/network objects remain in the host environment. Portable values are page records, updates, revisions and output descriptions. Never transport a DOM node as a supposedly portable natlang value.

## Crisp environments and authority

Use distinct environment bindings for trusted wiki application code and page-authored code. The application may directly share native host objects. A page programme receives the environment deliberately assigned to it; reading a page does not automatically provide access to the viewer's private resources. A trusted local page may explicitly use shared access. A public deployment chooses an appropriate isolated/mediated implementation where its authority contract requires one.

Illustrative meta-APIs load a pinned cell revision, validate its engine closure and start a child run. Browser inference is a separate embedding implementation with actual memory/operator/latency validation. If it cannot reproduce the agreed merge profile, use a declared supported merge host or preserve disagreement; do not pretend all browser backends match.

## Reduction and streams

A Fold per page/session receives edits, remote updates, run commands and result events. Transport arrival order and semantic merge presentation order are recorded separately. P02 chooses the merge meaning. A merged code block must pass loading/checks before becoming the active runnable revision; preserve both source alternatives when that fails.

Running cells retain their source/input revisions. Late results cannot overwrite a newer page's cell outputs. Offline branches reconnect through semantic history reconciliation; there is no requirement for live lambda instructions to change as collaborators type.

## Delivery and checks

1. One page with prose and one pure cell, two simulated clients. Gate: edits preserved, semantic merge quality checked, results attributed to pinned source.
2. Real transport, disconnect/reconnect and semantic conflicts. Gate: agreed-profile repeatability and visible unresolved alternatives.
3. Browser renderer/runtime adapter, then local inference. Gate: common trace fixtures readable and required execution contracts supported.
4. Add richer assets and selected server effects only through declared environments.

Test invalid merged code, mismatched profiles, stale completion, deleted cells, duplicate updates, private-data access attempts and unavailable engines.

## Trace and teacher

Link page revision, merge run and cell execution traces without conflating their guarantees. Train merge quality and programme execution separately. A shared-host cell can have diagnostic-only effect coverage; the UI must show it. Replay of a page's display must not rerun embedded effects. P02, C3 and C5 are meaningful dependencies; a universal sync or sandbox framework is not.
