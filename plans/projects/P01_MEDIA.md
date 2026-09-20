# P01 — Semantic media workbench

Status: CPU-backed batch workbench implemented in
[`codebases/media_workbench`](../../codebases/media_workbench/README.md) and
[`applications/media_workbench.mjs`](../../applications/media_workbench.mjs).
Interactive jobs, bounded revision and live-model quality remain open.
[Shared capability definitions and conventions](README.md). Product scope remains in the [catalogue](../AMBITIOUS_PROJECTS.md#p01-semantic-media-workbench).

## Natlang prerequisites

Use C0 calls, finite Map and bounded Iterate for planning, batch inspection and revision. C1 must permit a host-backed engine with suitable operation lifetime; the existing short effectful QuickJS deadline cannot be the general media execution policy. C4 records inspections and render outcomes. C5 is required only when requests, progress and cancellations coexist interactively. No binary primitive, job type or global vision tool is needed.

## Programme and typed boundary

`transform.nl(request, inputs, policy) -> MediaResult` calls `understand.nl`, `choose_recipe.nl`, `plan.nl`, `inspect.nl`, `assess.nl` and `revise.nl`. Exact siblings compile the filtergraph and check technical constraints. Keep subordinate helper families in companion folders.

The tree contains `ClipSummary`, `TransformPlan`, `Inspection`, and `MediaResult` records: ordinary IDs, dimensions, durations, stream metadata, requested constraints, observations and accepted output IDs. Native frames, buffers and job objects stay in the media eval environment. A result distinguishes verified output, unsupported request and failure/unknown completion; a filename alone is not evidence of a completed render.

## Crisp environment

Start with a host-backed `ts` binding exposing illustrative `media.probe(id)`, `media.render(plan)`, `media.sample(id, selection)` and `vision.inspect(samples, question)` APIs. Exact methods may return real native objects inside eval; snippets convert observations to portable records. If the actual host is Python, provide a suitable engine or explicit bridge rather than claiming direct JS/Python object sharing.

Use one environment per workspace/session. Retain jobs and previews there across eval calls; release temporary media after publication or cancellation. Inspection-model identity and sampled frame positions are recorded. A missing visual engine yields an unsupported observation. Natlang decides what to inspect and whether to revise; `media.render` only executes the supplied exact plan.

## Reduction and stream shape

The batch path is an ordinary call followed by bounded inspect/revise iterations. The interactive path is `Fold<MediaEvent, MediaSession>` over request, progress, completion and cancel events. A start snippet registers a native job and returns promptly; its completion later enters the same stream. Correlate job and request revision so late previews cannot satisfy a newer request. Do not make the model poll progress or accept mid-step instruction mutation.

## Delivery and checks

The batch implementation probes and hashes input/output media, executes real
FFmpeg trim/crop/scale/transcode commands, samples a frame for an optional
visual inspector, and independently checks dimensions, duration, audio and
receipt identity. A crop without visual evidence stays under review. Real
FFmpeg integration tests cover trim, crop, corrupt input, impossible geometry
and the visual callback boundary. Output collisions and interrupted renders
are explicit rather than silently reused or called successful. The current
scripted model tests do not establish recipe-selection quality; that needs a
teacher pilot and separate semantic review before training admission.

1. Probe/trim/crop/transcode a synthetic clip with exact metadata checks. Gate: expected streams/duration/resolution and independently verified output existence.
2. Add semantic recipe selection and visual inspection, then one bounded revision. Gate: technical checks plus a separate crop/intent rubric.
3. Add interactive stream, progress and cancel behavior. Gate: a late completion cannot overwrite the active revision; cancellation reports the actual observed outcome.

Fault fixtures include corrupt input, no audio, impossible target, missing model, timeout after output creation and stale native IDs. Stream support does not require durable recovery in this first interactive version.

## Trace, teacher and portability

Capture source/input identity, plan, eval engine/environment, sampled evidence and tool outcomes. Native mutations are not automatically replayable; use frozen observations and reset media fixtures for replay. Teacher samples cover selection, inspecting uncertainty, valid revision and honest failure. A browser host may expose a different media API implementation; programme compatibility requires matching helper contracts, not an FFmpeg installation on every natlang target.
