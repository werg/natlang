# S03 — Interfaces invented for the task

Status: proposed. Part of [the semantic software plan](../SEMANTIC_SOFTWARE.md).

## Product and semantic ambition

Natlang should devise an interaction that helps someone understand or change
the current situation. Choosing an existing dashboard panel is a useful start,
but the application should also create a new comparison, manipulation or
experiment and bind it to executable natlang behavior.

Example: an investigation has two competing explanations for a result. Natlang
builds a paired comparison with linked exclusions, a counterfactual control and
an evidence inspector. A control runs a generated analysis method and updates
the competing conclusions. The UI did not exist before this investigation.

## User experience

The notebook gains an “Explore” surface next to prose, cells and source. Natlang
can create an interaction there while preserving the user's chosen layout and
current work. Each generated view has a “How this works” inspector showing its
source, handler functions, bindings and relevant assumptions.

The UI stays usable while inference runs. Typing, focus, selection, scrolling,
pointer movement and visual interpolation are local browser operations. A
meaningful submission or settled manipulation emits an event to natlang. For
some tasks that means every discrete move; for others it means releasing a
slider or pressing “Compare.” Natlang chooses the task's interaction semantics;
the renderer implements the responsive mechanics.

Do not regenerate the interface on every model turn. Change it when the task
needs a new interaction, and show why a substantial structural revision helps.
Users may pin useful views and keep several alternatives side by side.

## Natlang program

- `design_interaction.nl(question, evidence, state)`: choose what action or
  comparison would help, its data and expected feedback.
- `build_view.nl(design, available_renderers)`: generate view source and typed
  event bindings, using existing components or generated HTML/JS/TS.
- `write_handler.nl(binding, workspace)`: create an ordinary natlang function
  that interprets the event, drives computations and returns domain updates.
- `review_interaction.nl(view, examples, observations)`: inspect behavior and
  accessibility findings, then correct the view when needed.
- `adapt_view.nl(old_view, new_findings, drafts)`: update the interaction without
  throwing away the user's ongoing work.

Handlers can invoke several host operations, generate methods and revise state.
They are not constrained to translating a click into a predeclared studio action.

## Two rendering paths, one event contract

1. **Typed view tree:** build on the existing optional `BrowserDomRenderer`.
   Use stable IDs, references to data and handler bindings. Extend it only for
   concrete cases such as linked table selection and editable comparisons.
2. **Generated module:** for a custom canvas, spatial editor or interaction that
   does not fit the tree, generate a versioned HTML/JS/TS module with explicit
   mount/update/dispose functions. It emits the same typed events.

The tree is a convenient standard library, not the complete set of UI ideas
natlang is allowed to express. Generated modules prevent a component catalogue
from becoming the new fixed domain-action menu.

Environment policy is independent: trusted local modules may share selected
native DOM/data objects. Content from an untrusted collaborative page needs an
appropriately restricted environment. Document which host owns the module;
do not describe shared host objects as a sandbox.

## Binding and lifecycle contract

`ViewRevision` references a source manifest, data/schema revision and controls.
Each `ControlBinding` records stable identity, event payload type, handler root
and handler manifest. An event carries view revision, control ID, base workspace
revision, payload and event ID. It cannot become an invocation of whichever
handler happens to have the same name after an update.

The host validates payload shape and binding identity, then dispatches to the
bound natlang program. Natlang decides the domain meaning of the event.

The renderer reconciles stable elements and preserves drafts, focus, selection
and scroll where identities persist. Removed or meaningfully changed controls
retain their drafts in a recoverable tray. A stale event is rebased explicitly
or returned with a useful explanation; silently applying it to a different
dataset is unacceptable.

On reload, mount the stored successful view before starting optional model
work. If a new view fails to compile or render, retain the previous functioning
view and expose diagnostics to natlang. Isolate candidate preview lifecycle
from the current UI so a failed experiment does not erase the workspace.

## Required changes and sequence

1. Audit `BrowserDomRenderer` against this contract. The renderer exists;
   version-bound bindings, draft reconciliation and generated modules are new
   work, not assumed existing capabilities.
2. Add one generated comparison tree whose handler is a new natlang function.
   Integrate it into the actual notebook instead of an isolated demo page.
3. Add view revision persistence, stale-event handling and local draft recovery.
4. Add generated module preview, compile diagnostics, disposal and lifecycle
   checks through the existing selected evaluator and worker facilities where
   applicable. Browser DOM rendering itself stays on its appropriate thread.
5. Implement a custom interaction that cannot be expressed by rearranging the
   existing studio panels, then reuse the event binding library in the wiki.

## Tests and teacher tasks

Exact browser checks: keyboard operation, labels, focus retention, mobile
layout, repeated mount/dispose, no duplicate listeners, typed event rejection,
schema/view mismatch, cancellation and reload with a half-written draft.
Use real DOM events and handlers. A screenshot alone is not interaction testing.

Semantic scenarios:

- Compare explanations by including/excluding a cohort, with denominator effects
  visible rather than concealed behind an attractive chart.
- Resolve ambiguous entity matches through a generated side-by-side editor.
- Create a counterexample playground for a newly learned function.
- Explore a spell's interpretation before applying it to a game world.
- Build a log timeline that exposes the evidence relevant to an incident.

Acceptance: a generated interaction absent from authored panels performs useful
work through natlang, accurately reflects its evidence and remains usable across
revisions. Assess whether users can resolve the actual task, not merely whether
the DOM is valid. Train on interaction design, execution feedback and repairs.

## Risks and boundaries

Watch for interface churn, hidden assumptions in visual encodings, inaccessible
novel controls and useful behavior lost during view regeneration. Keep semantic
design freedom while retaining inspectable bindings and tested lifecycle code.
Do not move inference onto every keystroke to make the UI appear “more natlang.”
The semantic program's ownership is demonstrated by what it creates and does.
