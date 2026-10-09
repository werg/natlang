/**
 * The definitions of the program improver whose recorded invocations the exporters turn into training rows, and what is
 * common to the exporters: how a row names its stage, how a stage is linked to the experiment it belonged to, and how a
 * row declares its supervision.
 *
 * Two generations of recorded runs exist and both stay exportable. The keys are the definition sources the trace manifest
 * records, so a row never hides which generation it came from:
 *
 * - `rewriteProgram/1`: one editor, `improveStep/rewriteProgram.nl` (diagnosis, hypothesis and edit in one call), run by
 *   the model-run step `improveStep.nl`.
 * - `stages/1`: `improveStep/diagnose.nl`, `hypothesize.nl`, `editSource.nl` and `editSourceStructural.nl`, the
 *   natural-language search policies, and the counterexample suggestion. The experiment around them is crisp
 *   (`improveStep/lifecycle.ts`), so there is no step trace: an experiment's outcome comes from the run's state history.
 */
import {isDeepStrictEqual} from 'node:util';

export const LEGACY = 'rewriteProgram/1', STAGES = 'stages/1';

/** shape: editor (a directory reducer over the source folder), function (a typed value), policy (a pluggable search decision), step (the retired model-run sequence). */
export const IMPROVER_DEFINITIONS = new Map([
 ['improveStep.nl', {stage: 'improveStep', generation: LEGACY, shape: 'step'}],
 ['improveStep/rewriteProgram.nl', {stage: 'rewriteProgram', generation: LEGACY, shape: 'editor'}],
 ['improveStep/editSource.nl', {stage: 'editSource', generation: STAGES, shape: 'editor', mode: 'instruction'}],
 ['improveStep/editSourceStructural.nl', {stage: 'editSourceStructural', generation: STAGES, shape: 'editor', mode: 'structural'}],
 ['improveStep/diagnose.nl', {stage: 'diagnose', generation: STAGES, shape: 'function'}],
 ['improveStep/hypothesize.nl', {stage: 'hypothesize', generation: STAGES, shape: 'function'}],
 ['counterexampleStep/suggestCounterexamples.nl', {stage: 'suggestCounterexamples', generation: STAGES, shape: 'function'}],
 ['improveStep/findOpportunity.nl', {stage: 'findOpportunity', generation: STAGES, shape: 'policy'}],
 ['improveStep/chooseParent.nl', {stage: 'chooseParent', generation: STAGES, shape: 'policy'}],
 ['improveStep/selectIncumbent.nl', {stage: 'selectIncumbent', generation: STAGES, shape: 'policy'}],
 ['improveStep/shouldStop.nl', {stage: 'shouldStop', generation: STAGES, shape: 'policy'}],
 ['componentSearchStep/chooseMove.nl', {stage: 'chooseMove', generation: STAGES, shape: 'policy'}],
 ['componentSearchStep/chooseComponents.nl', {stage: 'chooseComponents', generation: STAGES, shape: 'policy'}],
]);

/** Definitions that are recorded in improver traces but are not exported here, with the reason (a registered omission). */
export const OMITTED_DEFINITIONS = new Map([
 ['componentSearchStep.nl', 'model-run component-search step: its trajectories belong to the component-search exporter'],
 ['counterexampleStep.nl', 'model-run counterexample step: its trajectories belong to the counterexample exporter'],
 ['componentSearchStep/editComponents.nl', 'component edit: exported with the component-search trajectories, which carry the engine state it is checked against'],
]);

/** The stage record of a definition source; an unknown source is an error that lists the known ones. */
export function stageOf(definitionSource) {
 const found = IMPROVER_DEFINITIONS.get(definitionSource);
 if (!found) throw Error('Unexpected optimizer invocation: '+definitionSource+'. The exported definitions are '+[...IMPROVER_DEFINITIONS.keys()].join(', ')+'.');
 return {definition_source: definitionSource, ...found};
}
export const isTrainableDefinition = definitionSource => IMPROVER_DEFINITIONS.has(definitionSource);
export const omissionOf = definitionSource => OMITTED_DEFINITIONS.get(definitionSource);
export const editorSources = () => [...IMPROVER_DEFINITIONS].filter(([, stage]) => stage.shape === 'editor').map(([source]) => source);

/** The definition source a trace records in its manifest. */
export const definitionSourceOf = trace => trace.events?.find(event => event.kind === 'manifest')?.definition_source;

const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const redacted = value => isObject(value) && ('$diagnostic_preview' in value || value.complete === false);

/**
 * The exact arguments (by parameter name) and return value of an invocation. Complete host captures win; otherwise the
 * bounded state events are used when they retained complete values. A redacted value cannot be replayed, so it is an error.
 */
export function invocationFields(trace) {
 const captures = trace.events?.filter(event => event.kind === 'host_capture' && event.call_id === trace.callId) ?? [];
 // A capture names every argument of any captured definition; the ones this definition lacks are recorded as missing.
 const inputs = captures.filter(event => event.capture_kind === 'invocation_input' && event.reason !== 'missing'), output = captures.find(event => event.capture_kind === 'invocation_output' && event.name === 'return');
 if (inputs.length || output) {
  if (!output?.complete || inputs.some(event => !event.complete)) throw Error('Exact host capture is incomplete for '+trace.callId);
  return {args: Object.fromEntries(inputs.map(event => [event.name, event.value])), value: output.value, captured: true};
 }
 const initial = trace.events?.find(event => event.kind === 'state' && event.phase === 'initial')?.value?.$lambda;
 const final = trace.events?.findLast(event => event.kind === 'state' && event.phase === 'final')?.value?.$lambda;
 if (!initial || !final || !isObject(initial.args) || Object.values(initial.args).some(redacted) || redacted(final.return)) throw Error('Trace redacted the exact arguments or return of '+trace.callId+'; replay evidence is unavailable');
 return {args: initial.args, value: final.return, captured: false};
}

/**
 * The classes of a row's prompt messages under whole-trajectory supervision (the owner rule). Instructions and inputs
 * (everything before the first assistant reply) are trained at the context weight. Messages up to the last earlier
 * assistant reply are supervised by the rows of those replies. Later non-assistant messages are tool output and mechanical
 * feedback, trained at the lower feedback weight. Nothing is masked here; the trainer applies `--context-weight` and
 * `--feedback-weight` (training/neuralese/natlang_neuralese/serve/grad.py `_context_weights`).
 */
export function supervisionOf(messages) {
 const assistants = messages.flatMap((message, index) => message.role === 'assistant' ? [index] : []);
 const first = assistants[0], last = assistants.at(-1);
 const classes = messages.map((message, index) => assistants.length === 0 || index < first ? 'instructions-inputs' :
  index <= last ? 'earlier-turn' : 'feedback');
 return {scope: 'whole-trajectory', classes, masked: [],
  weights: {'instructions-inputs': 'context', 'earlier-turn': 'supervised by the row of that reply', feedback: 'feedback', target: 'target'},
  rule: 'Prompts, instructions and inputs are trained; tool output and mechanical feedback after the first reply are trained at the lower feedback weight; nothing is dropped from the context.'};
}

/** The row fields that name the stage, so that a row never hides which definition and generation it came from. */
export function stageFields(stage, extra = {}) {
 return {improver_stage: {stage: stage.stage, generation: stage.generation, definition_source: stage.definition_source, shape: stage.shape,
  ...(stage.mode ? {mode: stage.mode} : {}), ...extra}};
}

/**
 * Link the stage invocations of one run into experiments by the values they pass on: a hypothesis was made from a
 * diagnosis, an edit from a hypothesis. `invocations` are `{callId, stage, args, value}` in trace order. Returns, for each
 * invocation call id, the experiment it belongs to (`{diagnose, hypothesize, edit}` call ids; absent parts are undefined).
 */
export function linkExperiments(invocations) {
 const request = invocation => invocation.args.request ?? {};
 const claimed = new Set(), experiments = [], byCall = new Map();
 // Stages of one experiment run in sequence, and experiments do not overlap, so a stage belongs with the nearest earlier
 // unclaimed stage whose value it received (two experiments can produce equal values; order tells them apart).
 const earlier = (index, stage, value) => {
  for (let at = index - 1; at >= 0; at--) {
   const item = invocations[at];
   if (!claimed.has(item.callId) && item.stage.stage === stage && isDeepStrictEqual(item.value, value)) { claimed.add(item.callId); return item; }
  }
 };
 const position = callId => invocations.findIndex(item => item.callId === callId);
 for (const [index, edit] of invocations.entries()) {
  if (edit.stage.shape !== 'editor' || edit.stage.generation !== STAGES) continue;
  claimed.add(edit.callId);
  const hypothesis = earlier(index, 'hypothesize', request(edit).hypothesis);
  const diagnosis = hypothesis && earlier(position(hypothesis.callId), 'diagnose', request(hypothesis).diagnosis);
  experiments.push({edit: edit.callId, hypothesize: hypothesis?.callId, diagnose: diagnosis?.callId});
 }
 // A hypothesis of kind none ends its experiment without an edit.
 for (const [index, hypothesis] of invocations.entries()) {
  if (hypothesis.stage.stage !== 'hypothesize' || claimed.has(hypothesis.callId)) continue;
  claimed.add(hypothesis.callId);
  const diagnosis = earlier(index, 'diagnose', request(hypothesis).diagnosis);
  experiments.push({hypothesize: hypothesis.callId, diagnose: diagnosis?.callId});
 }
 for (const diagnosis of invocations.filter(item => item.stage.stage === 'diagnose' && !claimed.has(item.callId))) experiments.push({diagnose: diagnosis.callId});
 for (const experiment of experiments) for (const part of ['diagnose', 'hypothesize', 'edit']) if (experiment[part]) byCall.set(experiment[part], experiment);
 return byCall;
}

/**
 * Whether a stage's output may be a positive target, from the independently measured outcome of the experiment it
 * belonged to (an entry of the run's state history, which the host wrote from its own measurements):
 * - an edit: the entry whose candidate is the edit's resulting source is accepted; an unchanged result is honest;
 * - a hypothesis of kind none: the experiment made no edit, an honest result like an unchanged edit;
 * - a diagnosis or hypothesis that led to an edit: the edit was approved;
 * - a diagnosis with no hypothesis found, or an edit with no measured entry: context only.
 * Policies and the counterexample suggestion are admitted by their replay and the verifier that bounded them.
 */
export function stageAdmission({stage, invocation, experiment, outcomes, editApproval, shadowDisagreements = new Set()}) {
 if (stage.shape === 'policy' && shadowDisagreements.has(stage.stage)) return {approved: false, reason: 'A shadow run recorded a disagreement with the crisp decision that was served; the natural-language side is kept as context.'};
 if (stage.shape === 'policy') return {approved: true, reason: 'Exact recorded inputs and result independently reexecuted; the crisp verifier that bounded the choice accepted it.'};
 if (stage.stage === 'suggestCounterexamples') return {approved: true, reason: 'Exact recorded inputs and result independently reexecuted.'};
 if (stage.shape === 'editor') return editApproval;
 if (stage.stage === 'hypothesize' && invocation.value?.kind === 'none') return {approved: true, reason: 'Honest hypothesis of no edit: the experiment ended without changing the source.'};
 const approval = experiment?.edit ? outcomes.get(experiment.edit) : undefined;
 if (!approval) return {approved: false, reason: 'The experiment around this '+stage.stage+' has no measured edit outcome; kept as context.'};
 return {approved: approval.approved, reason: approval.reason};
}

/**
 * The admission of an edit under the stages generation, from the run's state history (the host's own measurements, since
 * no step invocation records them any more). An unchanged result is an honest one. The first history entry whose
 * `candidate` is the resulting source is the experiment that measured it (later entries with the same candidate are
 * `duplicate` outcomes). An edit with no entry, for example the experiment in which a run stopped, is context only.
 */
export function measuredEditAdmission(history, candidateDigest, unchanged) {
 if (unchanged) return {approved: true, reason: 'Honest unchanged editor result.'};
 const entry = (history ?? []).find(item => item.candidate === candidateDigest);
 if (!entry) return {approved: false, reason: 'No independently measured experiment in the run state for this edit; kept as context.'};
 return {approved: entry.accepted === true, reason: entry.reason};
}

/**
 * The services the model-run lifecycle step and its editor saw in their scope when a run of the rewriteProgram generation
 * was recorded: the evaluator, and the plan journal once the authored lifecycle took it. The current improver's stages see
 * no service (the crisp step holds them), so a replay of those registers none.
 */
export function legacyServices(authoredFiles, evaluatorDeclaration, plansDeclaration) {
 const journaled = /\bplans\b/.test(authoredFiles['improveStep/lifecycle.ts'] ?? '');
 return {services: {evaluator: {}, ...(journaled ? {plans: {}} : {})}, serviceDeclarations: {evaluator: evaluatorDeclaration, ...(journaled ? {plans: plansDeclaration} : {})},
  serviceScopes: {evaluator: ['improveStep.nl'], ...(journaled ? {plans: ['improveStep.nl']} : {})}};
}
