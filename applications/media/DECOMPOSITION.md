# Media: decomposition, part by part

Status: implemented 2026-10-09 on the owner's instruction to finish everything; review after the fact (plans/OWNER_REVIEW.md). "As built" at the end records where the build differs from or settles this draft.

`transform` (`index.ts:171-196`) probes a clip, lets natural language choose one plan, renders it with FFmpeg,
inspects the result, lets natural language assess intent, and derives the final status from exact checks. The
crisp half (probe, render, inspect, technical checks) is correct. The two natural-language functions are
too coarse: `choose.nl` selects the operation, computes parameters for four operations and decides audio in one
call (`choose.nl:8-15`), and `assess.nl` mixes a note read, an intent judgment and a visual-review flag
(`assess.nl:11-17`). The model also copies `input` and `output` back to the host, which then checks the copy
(`index.ts:179-180`).

Decisions: **fn**, **inline**, **implicit**, **crisp**, **service**, **host**, **pluggable** (as in the other
decompositions).

## Policy

- **Natural language: what the user meant.** Which operation, its parameters from the request and the source clip,
  whether the request depends on what the picture shows, and whether the inspected result meets the request.
- **Crisp: everything FFmpeg and everything exact.** The workspace boundary (`index.ts:66-81`), probe, render,
  the plan's validity rules (`index.ts:113-124`), inspection, the expected-metrics check (`index.ts:184-189`), and
  the status derived from those checks (`index.ts:192-193`).
- **The model never copies identifiers.** `input` and `output` come from the request; the crisp assembler writes
  them into the `Plan`. The check at `index.ts:179-180` disappears with the copy.
- **Retry once with the problem.** A plan that fails the crisp validity rules returns to the parameter function once
  with the problem text, as the games app does at its commit. Today it ends as a failed receipt.
- **State model.** `transform` is one decision-then-commit pass over an immutable request: snapshot (request,
  source clip), stages decide, effects (the render) are performed by the service, the result is derived. Derived
  values form a DAG: source, operation, parameters, plan, receipt, inspection, assessment, status.

## Parts

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| SHA-256 of a file | crisp | `sha256`, `index.ts:24-28` | Exact. |
| Spawn without a shell, bounded output, timeout | service | `run`, `index.ts:31-46` | The outside world. Output cap 8192 and timeout 120 s are named settings. |
| Media id to path, outside-workspace refusal, no overwrite | crisp | `MediaWorkspace.path`, `index.ts:66-81` | Exact boundary. |
| ffprobe of a clip: size, duration, audio, hash | service | `probe`, `index.ts:83-102` | The outside world. |
| Which operation: trim, crop, scale, transcode, unsupported | fn, decision | `plan/chooseOperation` | A finite judgment; today bundled with parameters. |
| The file the request names (sidecar note) | fn (shared) | `plan/readNote` | Sub-task repeated in `choose.nl:9-10` and `assess.nl:12-13`. Built-in after N1. |
| Trim parameters: start and end seconds | fn | `plan/planTrim` | Time arithmetic against the source duration ("last five seconds"). |
| Crop parameters: x, y, width, height | fn | `plan/planCrop` | Rectangle arithmetic against the frame. |
| Scale parameters: width, height | fn | `plan/planScale` | Aspect ratio arithmetic ("half size", "720p"). |
| Transcode: no parameters in the plan | crisp for now | `plan/assemble` | See question 1: the plan has no codec or container field. |
| Keep audio | fn, decision | `plan/keepAudio` | One yes or no from the request and `source.has_audio`. |
| Assemble `Plan`: kind, parameters, request.input, request.output, zeros for unused fields | crisp | `plan/assemble` | Removes "Copy request.input and request.output exactly" and "Use zero for unused numeric fields" (`choose.nl:11-13`). |
| Plan validity: finite numbers, trim interval inside duration, crop rectangle inside frame, positive integer scale | crisp | `checkPlan`, from `index.ts:113-124` | Exact; moved before the render so its problem text can return to the stage. |
| FFmpeg argv, codecs, preset | service | `render`, `index.ts:125-138` | Mechanism; fixed codecs are a setting. |
| Inspect the output: probe plus a sampled frame for crops | service | `inspect`, `index.ts:141-165` | The outside world. |
| Visual judgment of a crop frame | service | `VisionInspector` callback, `index.ts:21-22`, `155` | A multimodal model call outside the text runtime; it stays a service until the runtime accepts images. |
| Does the request depend on where the subject sits in the picture | fn, decision | `assess/needsPicture` | A sub-task of `assess.nl:14-16`. |
| Does the inspected result meet the request | fn | `assess/assessIntent` | The semantic core of the assessment; skipped when the technical check fails. |
| Expected width, height, duration, audio | crisp | `index.ts:184-186` | Arithmetic over the plan and the source. |
| Technical check including the 0.16 s duration tolerance | crisp | `index.ts:187-189` | Exact verifier; the tolerance becomes a named setting (encoder frame rounding). |
| Needs visual review | crisp over `needsPicture` | `index.ts:190` | A crop with an unsupported frame, or a picture-dependent request with an unavailable or uncertain frame. |
| Final status ladder | crisp | `index.ts:192-193` | A pure function of verifier outputs and the assessment. Every rung is a combination of exact facts. |
| Explanation text | crisp selection | `index.ts:195` | Chooses between the assessment and the technical failure. |
| Event log drain | host | `drainEvents`, `index.ts:167` | Mechanism. |

## Natural-language functions, step by step

### `plan/chooseOperation`

```
args: request: MediaRequest, source: Clip, note: Untrusted<string>
returns: "trim" | "crop" | "scale" | "transcode" | "unsupported"
```

1. Read `request.text` and `note`; name the change the user wants in one phrase.
2. Match it to exactly one of: trim (shorten in time), crop (cut the frame), scale (change the pixel size),
   transcode (re-encode without those changes).
3. When the change is none of these, or needs two of them, answer "unsupported".

### `plan/planTrim`

```
args: request, source, note
returns: { start: Is<number, "at least 0 and below end">, end: Is<number, "above start and at most source.duration"> }
```

1. Read the times in the request; convert to seconds (minutes times 60).
2. For "first N seconds": start 0, end N. For "last N seconds": start `source.duration - N`, end `source.duration`.
   For "from A to B": start A, end B.
3. Clamp end to `source.duration` and make sure start is below end.

Do the subtraction and clamping in eval.

### `plan/planCrop`

```
args: request, source, note
returns: { x, y, width, height: Is<number, "whole numbers; the rectangle lies inside the source frame"> }
```

1. Read the target size or region in the request. A named region ("center", "left half") becomes a rectangle:
   center of `W x H` for size `w x h` is `x = (W - w) / 2`, `y = (H - h) / 2`; halves and quarters use `W / 2`,
   `H / 2`.
2. Round x, y, width and height down to whole numbers.
3. Check `x + width <= source.width` and `y + height <= source.height`; shrink width and height to fit.

### `plan/planScale`

```
args: request, source, note
returns: { width, height: Is<number, "positive whole numbers"> }
```

1. Read the target: absolute size ("1280x720"), one side ("width 640"), a factor ("half"), or a named height
   ("720p").
2. With one side or a factor, compute the other side as `source.width / source.height` times the given side; round
   to whole numbers.
3. Return width and height.

### `plan/keepAudio`

```
args: request, source
returns: boolean
```

1. When `source.has_audio` is false, answer false.
2. When the request asks to remove, mute or drop the sound, answer false.
3. Answer true.

### `assess/needsPicture`

```
args: request
returns: boolean
```

1. Decide whether the request asks for a subject, face, object or composition to be in a place in the frame.
2. Answer true for such requests and false for requests about time, size or format only.

### `assess/assessIntent`

```
args: request, source, plan, observed: { width, height, duration, has_audio, visual_status, visual_detail }, note
returns: { intent_met: boolean, explanation: Is<string, "cites the observed values it relies on"> }
```

1. Restate what the request asks in one phrase, using `note` when present.
2. Compare it with `plan` and the observed values; the observed values are the evidence, not the output file name.
3. When `visual_status` is `contradicted`, answer intent_met false.
4. Write the explanation in one or two sentences naming the observed values used.

`assessIntent` runs only when the crisp technical check passed; otherwise intent_met is false and the explanation is
the technical failure (`index.ts:195`).

## Refinement candidates

| Slot | Proposed type | Check |
| --- | --- | --- |
| `Plan` (trim) | `Is<Plan, "start >= 0, start < end, end <= source.duration + 0.02">` | crisp (`index.ts:117`) |
| `Plan` (crop) | `Is<Plan, "x, y, width, height are whole numbers and the rectangle lies inside the source frame">` | crisp (`index.ts:119-122`) |
| `Plan` (scale) | `Is<Plan, "width and height are positive whole numbers">` | crisp (`index.ts:123`) |
| `Plan.keep_audio` | `Is<boolean, "false whenever the source has no audio">` | crisp |
| `Assessment.intent_met` | `Is<boolean, "false whenever the inspection did not succeed">` | crisp; replaces "If technical inspection failed, set intent_met false" (`assess.nl:13-14`) |
| `Assessment.needs_visual_review` | `Is<boolean, "true for a crop whose visual status is not supported">` | crisp; replaces `assess.nl:14-15` and half of `index.ts:190` |
| `Assessment.explanation` | `Is<string, "cites the observed values it relies on">` | judged |
| `MediaRequest.text`, sidecar note | `Untrusted<string>` | crisp marking |
| `MediaResult.status` | closed union (already) | crisp |

## Model-facing changes needing live measurement

1. **Split `choose.nl`** into `chooseOperation`, three `plan*` functions and `keepAudio`. Measure plan validity on
   the cases in `ts-host/test/media-workbench.test.mjs` and on recorded requests, especially trim ("last N
   seconds") and crop ("center") arithmetic.
2. **Remove the copy instruction** ("Copy request.input and request.output exactly", "Use zero for unused numeric
   fields", `choose.nl:11-13`); the model no longer sees the fields.
3. **Retry text.** The one-sentence problem returned to a parameter function after a failed `checkPlan`. Measure
   the repair rate on first retry.
4. **Replace the guards** "Do not claim to understand an unsupported operation by silently selecting a different
   one" (`choose.nl:14-15`) with the "unsupported" step and the closed enum; replace "A contradictory visual
   inspection must not be called successful" (`assess.nl:16`) with step 3 of `assessIntent`.
5. **Skip `assess` when the technical check fails.** One fewer model call per failed render; wording of the
   failure explanation is unchanged.
6. **Drop `output_sha256` and file names from the model-visible inspection.** The assessment sees observed values
   only. Measure that assessments do not lose evidence.

## Questions for the owner, as resolved in the build

1. **Codec and container: added.** `Plan` has `container` (`mp4`, `mkv`, `webm`, `mov`) and `video_codec` (`h264`,
   `h265`, `vp9`, `av1`), the closed set `MEDIA_CONTAINERS` / `MEDIA_ENCODERS` in `index.ts` supports. Audio codec follows
   the container (aac; libopus for webm). The service forces the muxer with `-f`, and `checkPlan` requires that the
   container holds the codec and that the output name carries the container's extension. Plans of the other operations get
   the container of the output name and its usual codec, so `trim` to `out.webm` encodes VP9 instead of failing. A plan given to
   `render` without the two fields gets the same defaults (older callers keep working). The technical check also compares the
   probed codec name with the planned encoder's.
2. **Duration tolerance: a named setting.** `DURATION_TOLERANCE_SECONDS` (0.16) is the default of the `durationTolerance`
   option of `MediaWorkspace`; the FFmpeg output cap is the `maxOutput` option (8192) beside `timeoutMs`.
3. **`VisionInspector` stays a service callback** until the runtime passes images to a model.

## As built

- Files: `chooseOperation.nl`, `planTrim.nl`, `planCrop.nl`, `planScale.nl`, `planTranscode.nl`, `keepAudio.nl`,
  `needsPicture.nl`, `assessIntent.nl`; `choose.nl` and `assess.nl` are gone. The note read is the runtime built-in
  `readNote` (`ts-host/src/builtin/readNote.nl`, `builtin('readNote')` exported from `@natlang/node`), shared with the
  other apps.
- `transform` = probe, `planRequest` (stages, `assemblePlan`, `checkPlan`, one retry), `render`, `inspect`, then the
  semantic stages only when the exact checks passed. A failed or unsupported render costs no assessment call.
- The retry goes to the parameter function with `problem` (the text of `checkPlan`). Problems about the output name
  (a non-transcode operation) do not return to a stage: no stage can change the name. Even width and height for crop and
  scale (the yuv420p format) are part of `checkPlan`, so an odd size returns to the stage instead of failing in FFmpeg.
- Stages read `SourceFacts`, `PlanFacts` and `Observed` (measurements and parameters), and the request text; file names and
  hashes are not model-visible.
- The `Is<...>` candidates for the plan are exact rules over the source clip, so they are `checkPlan` (with its problem
  text as the teaching error) rather than value-only refinement types, which cannot see the source. `intent_met` false on a
  failed inspection and the crop-review rule are crisp in `transform`. Only the sidecar note (file content) is
  `Untrusted<string>`; the request text is the user's own instruction and stays a plain string.
