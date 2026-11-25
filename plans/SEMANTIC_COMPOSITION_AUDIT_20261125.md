# Semantic composition: availability and actual behavior

## Mechanism inventory

The V27 catalog selects `entry.program` and `entry.callable`; System-testing's
`workers/system-test-runner/authored-teacher.ts` loads and invokes that exact entry.
Base `ProgramLoader` links the assigned workspace's installed modules and prunes
unavailable branches. Core `runtime/loader.ts` binds companion folders and explicit
`uses`; `runtime/callable.ts` constructs those callable trees; `runtime/hooks.ts`
passes their scopes into inline calls. `native/agent.ts` renders the actual scope.
Node, browser and Workerd share this mechanism; managed llama, wllama, Python
native and SGLang consume model requests, not separate application binders.
Collectors and trajectory IR preserve actual model decisions and crisp calls;
crisp call-tree edges are not model-produced learning targets.

The authoring cue appeared in Base's installed `natlang-authoring/SKILL.md`
(algorithm/decomposition fixed in source, model takes the next step), and Core's
authoring/integration introductions emphasized small judgments with TS doing the
rest. These guides and Core's composition reference are corrected together. Pure
value-producing leaves, exact effects and authored scope authority stay valid.

## Frozen availability census

Inputs: V27's original 2,327-row catalog and eight template sources (Base/SST
shared providers plus the six assigned workspace families). The actual installed
linker produced 33 program-directory views. All 367 entry/script-imported NL binding instances and their bound callable
trees loaded without errors. A preliminary raw-file scan's missing-mount type
errors were artifacts of that scan; they are not production defects.

| Entry execution kind and reachable scope | Workspace/entry identities | Catalog tasks |
| --- | ---: | ---: |
| Agent entry with authored helpers | 13 | 175 |
| Agent entry without authored helpers | 1 | 1 |
| Crisp TS entry with an NL subroutine exposing helpers | 94 | 1,936 |
| Crisp TS entry reaching only helper-less NL functions | 10 | 215 |

These are availability counts, not observed delegation, quality, or training-token
shares. A TS entry can hand the full goal to a semantic composer. A helper-less NL
function can use admitted native services or create inline NL functions. For
example, board migration's NL functions directly inspect and retain through scoped
services despite having no authored callable children. Do not classify them as
pure judgments from the folder structure.

Concrete fixed-coordinator examples include `reviewWorkspaceClaims` (selection,
packet/evidence reads, support judgment, summary, optional retained note),
`workInBrowser` (an action enum controls a TS loop), and
`coordinatePersonalAutomation` (a plan value controls a TS loop). Their policy
judgments are useful; these roots do not themselves train model-owned domain
call selection. Much of the corpus also exposes meaningful nested semantic
operations below a crisp root, so the task totals cannot substitute for tracing
which of those operations the model actually chooses.

## Two original observations

V27 wave0055, `open-goal-vault-graph-draft-seat-count-and-signed-limit`: eight
recorded native calls comprise one crisp root with zero model turns, six
`chooseKnowledgeStep` calls whose openings contain no authored callable section,
and one inline answer. TS performed the navigation; the model returned step
values. The original native checks failed; this audit does not attest success or
change that assessment. Spectrolite's separate feedback repair belongs to its
application owner.

Wave0047, `open-goal-verify-and-correct-tutorial`: the agent
`reviseFromPublicResearch` parent used five model turns and invoked named children
for actual document context, a research question, public investigation and document
revision. `researchBrief` invoked `discoverSources` and `brief`; the revision
function invoked its own policy children. Sixteen native calls, including inline
semantic questions, show real model-owned composition in that episode. This is
evidence for that run, not a required route for other goals or a corpus-wide
behavioral estimate. The original capture is unchanged.

## Acted on and next evidence

The shared guidance now teaches domain-specific semantic ownership of open-ended
decomposition, hierarchical access to NL/TS operations, exact TS effects and
legitimate pure policy leaves. The working tutorial is the concrete example. No
filename rule, automatic scope widening, mechanical rewrite of 104 TS roots, new
task world or corpus/data deletion follows from the census. The runtime scope
repairs shipped separately in Core70ed.

Next collection analysis uses real traces: bind child calls to the parent's actual
model action and callable scope, distinguish model-selected domain delegation
from crisp orchestration, and measure the resulting learning-target/token shares.
Availability is not a qualification gate. More original episodes/domains and
actual training-target weights remain unchecked here; no paid probe was run.
