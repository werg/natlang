# Visual, frontend and 3D self-improvement expansion

Researched 2026-10-04. **Status: source intake integrated / artifact executors pending.** No new episodes, model runs or admitted training tokens are claimed. Proposed families are registered in `training/self_improvement_tasks.json` so this work stays visible in the pipeline inventory.

## First collection priorities

These are proposed project-generated tasks, with requirements supplied in text. They can start without image-conditioned inference; screenshots remain host evidence until the provider, runtime and training serialization have an audited image path.

| Order | Family | Example task and improvement landscape | Independent checks |
|---|---|---|---|
| 1 | Responsive HTML/CSS repair | Repair a product comparison, multilingual booking form, timetable or dense dashboard with overflowing cards, unreadable labels and broken small-screen layout. | Pinned browser measures overflow, text visibility, clipping, target size, reading order and declared alignment at multiple viewport sizes; exact content and behavior checks remain mandatory. |
| 1 | Frontend JS state and interaction repair | Accessible filterable catalog; editable invoice table; wizard with dependencies, validation, undo and persistence; async search with stale-response races. | Host browser drives actions and compares resulting visible state to an independent state machine. Cover keyboard, focus restoration, invalid inputs and adversarial event order. |
| 1 | Semantic SVG diagrams | Render a dependency graph, evacuation map, process diagram or annotated time series while preserving directed edges, labels, grouping and legends. Improve crossings, label placement and hierarchy. | Parse SVG and verify semantic entities/edges plus rendered positions, clipping and text boxes. Score readability constraints gradually; do not require one exact arrangement. |
| 2 | SVG repair and compaction | Fix clipped arrows, broken viewBox, tiny labels, wrong layering or missing contrast; simplify an icon without losing features at several sizes. | Render at several scales, check required geometry/labels and visual tolerances. Byte/path count is secondary after content and visual gates. |
| 2 | Parametric 3D generation and editing | Design a cable organizer, bracket, nesting tray or vented enclosure from prose and dimensions. Edit hole spacing without losing clearances or wall thickness. | CadQuery/build123d or OpenSCAD executor checks compilation, units, dimensions, solid validity, holes, connectivity, minimum thickness and mating clearances. Tolerance is declared per property. |
| 2 | Mesh repair and simplification | Repair flipped normals, duplicate vertices and non-manifold seams; reduce faces while preserving silhouette and explicit features. | Independently recompute manifoldness, orientation, components, geometric deviation and feature preservation. Evaluate several views; face count only after validity. |
| 3 | Three.js / Blender scene construction | Produce an explainer scene with correct object relationships, camera framing, labels, lighting and an animation that demonstrates assembly or motion. | Host extracts scene graph and transforms, verifies relationships, renders fixed views/frames, checks occlusion and continuity. Artistic preference remains a separately calibrated semantic signal. |
| 3 | Screenshot-to-frontend reconstruction | Reconstruct or edit a UI from a reference image and specification, preserving responsive behavior. | Image-input support required. Use content, layout and behavior checks alongside visual comparison; one screenshot is insufficient for responsiveness or interaction correctness. |

Semantic complexity should come from user intent: what information belongs together, what must remain visible, what state an action changes, what a diagram communicates, and how a part fits another part. Random CSS property matching and primitive-counting alone will not supply that landscape.

## Primary source shortlist

External sources below are research candidates, not approved imports. Pin revisions, review source-level rights and record hashes before preparing cases. Keep existing benchmark test sets separate from any claimed independent evaluation; if repurposed as training, record that decision explicitly.

- [DesignBench](https://github.com/WebPAI/DesignBench) supports frontend generation, editing, repair and compile-error repair across vanilla HTML, React, Vue and Angular. Code-only edit/repair modes are particularly relevant to our initial text-based executor. License/provenance review is pending; no corpus copied.
- [Design2Code](https://github.com/NoviScl/Design2Code) provides real-page screenshot reconstruction and fine-grained automatic layout/text/color metrics. Its README has research-use language and C4/ODC attribution; separate code/data terms require review. Use its metric decomposition as a reference, with no automatic training admission.
- [WebSight](https://huggingface.co/datasets/HuggingFaceM4/WebSight) supplies synthetic HTML/screenshot pairs. Its card declares CC-BY-4.0, requires respecting source-content licenses, and disclosure of dataset use for released models/applications. Candidate for source-backed UI repair mutations or static pairs after asset and provenance audit; pairs alone are not improvement trajectories.
- [WebCode2M](https://huggingface.co/datasets/xcodemind/webcode2m/blob/main/README.md) declares CC-BY-4.0 and supplies webpage design/code/layout data. Real-web assets and upstream rights need individual review. Start with a bounded subset, not a bulk download.
- [StarVector / SVG-Bench](https://github.com/joanrod/star-vector) covers image-to-SVG, text-to-SVG and diagram generation, and cautions against pixel MSE as the sole metric. Code is Apache-2.0; the datasets combine sources and need separate license/provenance decisions. Prefer original generated diagrams first.
- [CADTestBench](https://github.com/dimitrismallis/CADTestBench) is especially well matched: executable geometric/topological predicates assess prompt requirements and emit failure diagnostics. It derives 200 CAD programs from CADPrompt, with abstract and detailed prompts; the repository declares MIT. Audit the dataset and CADPrompt lineage independently before import. Mutation analysis is a useful verifier-validation technique.
- [CADCodeVerify / CADPrompt](https://github.com/Kamel773/CAD_Code_Generation) studies iterative rendered-design verification and refinement. Candidate method reference and source of annotated text/CAD pairs; neither pairs nor final baseline outputs establish a replayable multi-step improvement history.
- [cadgenbench](https://github.com/huggingface/cadgenbench) covers CAD generation/editing with validity, shape, interface and topology metrics, plus an iterative build123d baseline. Interface constraints complement global shape similarity. Review dataset licenses and actual stored iteration artifacts before attempting trajectory conversion.
- [MiniWoB++](https://github.com/Farama-Foundation/miniwob-plusplus) and [VisualWebArena](https://github.com/web-arena-x/visualwebarena) are references for browser interaction environments. They measure operating a UI, rather than implementing one; reuse applicable interaction patterns in original frontend requirements, with separate provenance for copied material.

## Integration into our skill improvement loop

Existing `SkillAuthoringOptions` in `ts-host/src/improvement/skill-authoring.ts` accepts host-owned `executeCase`, `scoring`, and frozen descriptors. `SourceCaseExecution` / `SourceResultScoring` in `host.ts` provide the extension seam. `playwright-core` is already a dependency; that does not establish an installed browser or a complete sandboxed evaluator.

1. A fixed `.nl` entry accepts the visible task and returns a typed artifact file map. HTML/CSS/JS/SVG/CAD artifacts are outputs; the skill library is the variable that authoring improves. Artifact repair inside an execution is allowed only through declared tools and captured traces.
2. The host materializes outputs in a fresh sandbox, runs a pinned browser or CAD renderer, and computes measurements using immutable host checks. Candidate code never owns the test driver, oracle or score. Disable external network, keep secrets/data mounts absent, bound resources and clean workers. Compilation/render failure caused by the candidate differs from missing dependencies or host exhaustion; the latter is unscored infrastructure failure.
3. Support feedback supplies actionable measured violations, not reference code or hidden solutions. Examples: which region overflows and by how much; which action sequence leaves stale state; which hole is misplaced and its displacement. Public requirements remain visible. Query/transfer measurements stay sealed until the edited skills freeze.
4. Use several related tasks per support group, then hold out component families, interaction models, diagram grammars or part topologies for query/transfer. Different colors or dimensions of the same template are insufficient independence.
5. Run body-edit, description-edit and combined arms, then ablations. Candidate skill topics include responsive layout, focus/state handling, diagram routing, SVG coordinate systems, solid modeling and mesh topology. Catalog descriptions should distinguish triggers: a CAD solid-validity skill should not be offered as the default for decorative Three.js scenes. Keep soft description values supported through the existing skill metadata path.
6. Admit only replayable, independently measured gains. Retain failed attempts and paired outcomes with exact artifact/runtime/skill hashes; infrastructure failures do not become preference negatives. A trajectory adapter requires actual intermediate observations and edits—do not fabricate them from before/after images or final code.

## Scoring and pitfalls

Report a vector of semantic, functional, visual and efficiency scores, with a declared task-specific primary objective. Basic artifact validity gates scoring; partial compliance among otherwise valid outputs gives the gradual landscape. Strict success still requires all mandatory requirements. Size/latency cannot compensate for wrong behavior, missing content or invalid geometry.

For UI behavior tasks, use passed requirement groups as the primary bounded score. For layout/diagram tasks, combine only predeclared normalized violation measures, retaining each raw measurement. For CAD, count satisfied requirements on valid solids and preserve exact dimensional tolerances. For style/aesthetic tasks, independent judges are secondary and require calibration; no exact numerical aesthetic gold is presumed.

Prevent screenshot shortcuts: embedded reference rasters, hidden duplicate text, fixed absolute layouts that pass one viewport, and assets unavailable at replay. Verify text is visible in the rendered page; DOM presence alone is insufficient. For SVG, disable scripts/external resources and raster embedding where the task requires vector output. For CAD, bounding box/volume alone cannot prove hole orientation, wall thickness or mating correctness. For timing objectives, isolate contention and record environment identity; rendering can initially use CPU to protect training/inference GPU capacity.

Pin browser/renderer version, fonts, assets, viewport, device scale, locale, timezone and random seeds. Disable animations for static UI checks; capture explicit sampled frames for animation tasks. Separate robust geometry/content metrics from anti-aliasing-sensitive pixels. Text-only tasks can use numerical render diagnostics now; genuinely visual conditioning requires an audited image representation through inference, replay and training. That end-to-end capability has not been established by this research pass.

## Next concrete work

Implement the shared isolated artifact executor and two small project-generated pilots: responsive layout repair and semantic SVG diagrams. Validate verifiers with known-good outputs and deliberate mutations before model collection. Add frontend state-machine tasks and constrained CAD once their separate execution environments work. Measure baseline headroom on support, then collect skill-editing runs only where there is room for transferable improvement. Keep all eight families executor-pending until those checks are actually implemented.

## Integrated source intake (2026-10-04)

The source-intake layer is implemented. It is distinct from the browser/CAD
execution and collector layers, which remain pending.

- `training/visual_sources.json` locks nine acquisition lanes to immutable HF
  revisions, dataset-card hashes and exact selected files. Six code/method
  references have separate pinned revisions. SVG-Bench's unavailable endpoint
  and CADPrompt's manual gate/missing upstream license are explicit holds.
- `scripts/acquire_visual_sources.py` uses the stored HF login when available,
  bounded streaming transfers, delayed exponential retries, durable per-file
  receipts and atomic writes. It never runs remote loading scripts or source
  code. Raw files live under `vendor/datasets/visual-frontend/` and are covered
  by the existing development mirror.
- `scripts/prepare_visual_source_tasks.py` adapts source records into
  `natlang.artifact-source-task/1` packets, preserving visible inputs separately
  from host references/checks. It supports vanilla and React/Vue/Angular edits
  and compiler repair, responsive HTML adaptation, screenshot reconstruction,
  SVG refactoring, CADTestBench prompt/check joins and CADGenBench public inputs.
  Serialized artifacts and CAD joins are bounded. Prepared directories include
  frozen adapter/helper sources, source assets, oracles and a hashed held-row log.
- `scripts/audit_visual_source_tasks.py` verifies preparation identities,
  source/check separation, source-program groups, frozen helper hashes and held
  dispositions. It explicitly does not validate semantic reward or source rights.
- `scripts/run_visual_source_intake.py` connects acquisition, preparation, audit
  and registration. `training/self_improvement_tasks.json` has an
  `artifact_sources` lane, and the inventory reports its counts separately from
  executable native episodes. `training/data_sources.json` explicitly classifies
  source packets as awaiting evaluation, never as direct SFT inputs.

Current intake-v3: **734 packets / 534 source group IDs / 9 lanes (before cross-source content review)**, preparation
audit and central inventory have zero errors. Of these, 51 WebSight rows have
only the artifact-executor blocker; 683 rows have additional source/evaluator
holds. No packet is assigned a train/validation role: global cross-source
leakage review and grouping remain required. CADTestBench's 400 prompt variants
share 200 source-program groups. Native prepared targets remain 1,050 episodes /
7,813 problem instances; the artifact packets add zero active collector episodes
and zero admitted trajectories.

| Lane | Packets | Main remaining work |
|---|---:|---|
| WebSight v0.1 | 64 | Independent browser evaluator; 13 rows also need asset resolution |
| WebCode2M | 32 | Upstream rights/assets and browser evaluation |
| DesignBench vanilla edits | 32 | Missing dataset license and requirement/behavior evaluator |
| DesignBench React/Vue/Angular edits + compile repair | 48 | Dataset license, pinned framework dependency environments and evaluator |
| Design2Code | 16 | Research/source terms, image path, assets and evaluator |
| SVG-Diagrams | 29 | Dataset license and multiscale render/feature checks; 3 unsupported source rows retained separately |
| SVG-Stack | 32 | Dataset license and multiscale render/feature checks |
| CADTestBench | 400 | CADPrompt lineage rights, requirement/oracle mutation review and isolated CadQuery executor |
| CADGenBench | 81 | Image/STEP input path; private quality targets unavailable |

The source download is about 1.6 GB, rather than a bulk download of the large
web/SVG corpora. Both raw and prepared data have been mirrored to the DGX.
`intake-v1` and `intake-v2` are retained as superseded preparation drafts.
v1 had provisional source-local split labels; v2 mapped Angular template files
to incompatible names. v3 preserves `new.component.*` names and explicitly holds
16 Angular rows with source CSS references whose files are not supplied. Do not
use superseded packets for collection or training. The task registry retains
the former packet paths and hashes in `artifact_source_history`.

Run from the repository root with the Parquet-capable environment:

```sh
.venv/bin/python scripts/run_visual_source_intake.py --acquire
python3 scripts/self_improvement_task_inventory.py
```

An existing prepared output is audited, not rewritten. Use a fresh `--out` for a
changed adapter contract. Source spec changes also need a separate raw output
root, since acquisition rejects a changed lock at the same identity. Errors are
reported per source and do not become model failures. Bind only the declared
visible files into a future execution sandbox: `source_assets_root` also holds
oracle-bearing source files and must never be mounted wholesale for the model.

Benchmark sources are candidate curriculum material here; preserve their
original evaluation provenance and do not claim independent benchmark results
on examples subsequently used for training. No direct multi-step trajectories
were found in the inspected CAD sources. Before/after edits and final baselines
remain pairs/reference artifacts, not invented improvement histories.

## Static browser pilot — 2026-10-04

Implemented CPU-only Docker measurement plus a host-owned pilot reward in
`scripts/visual-browser/` and `ts-host/src/skills/visual-objective.ts`. The browser
gets candidate HTML only, never oracle/source directories. Chromium sandbox,
nonroot user, pinned official seccomp, no network, no host mounts/GPU, resource
bounds and infrastructure-error propagation are active. Retaining only
`SYS_CHROOT` fixes the sandbox startup failure without disabling its isolation.

Frozen local image ID:
`sha256:135269fa0c006f8fa7f7a01193e9437060bce240b586425a59acb620c0b7f471`.
This ID is architecture-specific; an ARM build needs its own recorded identity.
The multiarch base digest and matching Playwright 1.63.0 package/browser are pinned.

Provider-free `visual-browser-pilot/render-v6/report.json` screens all 51 eligible
WebSight source packets: **39 measured, 12 unsupported/unscored, 7 with baseline
headroom**. The remaining 32 measured pages have no headroom under this objective.
Headroom includes two cases whose narrow-view correctness gates fail; this is
not seven demonstrated model improvements. The held sources include ambiguous
paint/visibility and generated pseudo-element content. Baselines are not training
trajectories. All 21 checks in `mutations-v6.json` pass, including whitespace
equivalence, a genuine responsive repair, transparent/offscreen/covered text,
pseudo overlays, stripped design, changed links and external assets.

Earlier pilot iterations are retained: initial sandbox startup failed, early
reward mutations exposed concealed text and design stripping, and render-v3's
scorer was rebuilt while the screen was running. Its reward/hash association is
superseded; use v6, which records an unchanged before/after scorer hash. Do not
register older measurements as native cases.

This is still **held for production collection**. Desktop text paint/size and rough
geometry guards are conservative and do not prove complete aesthetic fidelity.
The pilot metric now dispatches through the central scorer and is code-pinned by
the headroom screen and collector. Its host expected task pins the immutable
image ID; an unavailable image remains an infrastructure failure. Training export
explicitly rejects this pilot metric, even if exploratory records are later collected.

Next: broader paint/decorative-region mutation review, reviewed whole-topic/template
role assignment, native skill episodes (body and description tuning variants),
reviewed admission contract, a fresh sealed runtime,
support headroom screening and actual verified collection. Static affordances are
the first contract; no functional JS interaction claim. Existing active queues and
frozen runtimes were not changed. New training admissions: zero.

An empty zero-size link exposed an affordance-scoring bug, now fixed. Real visible
links still fail preservation if hidden or changed. v6 has five positive-score
headroom cases and two narrow-view gate failures; the wellness failure has a
handwritten, fully scored CSS repair proof (quality1/all gates), kept host-only.
This is a verifier test, not an invented teacher/student trajectory.

Role review also found six cross-role alias edges in the first reconciled proposal.
`scripts/close_visual_source_roles.py` now unions every declared overlap and whole
cluster before assigning roles. `episode-proposal-v3.json` preserves all51members
in13closed components with zero crossing edges. Unsupported members remain held
within their component role; valid peers need not be discarded. These role labels
are review-only and have not been applied to the source registry or active queues.
