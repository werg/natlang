// natlang ops: the retained part of the application. One definition shared by the native participant (DGX) and the browser Worker.
// Retained facts are small and human-made: training-data edits (flag/exclude/annotate/correct), the native participant's record of
// applying each edit to the errata/exclusion/annotation layers, and questions to / answers from the supervising agent.
// Live ops state and training data are NOT retained here: the native participant serves them on demand over session signals
// (ops-protocol.mjs), so the signed journal never grows with 30 s snapshots or GB-scale rollouts.
// Changing these semantics requires a new definition version and a deliberate new application setup (the descriptor binds it).
import { defineApplication, defineCommand, defineQuery, emit, program } from '@chronolog/peer-session';

const str = (max, min = 0) => ({ type: 'string', minLength: min, maxLength: max });
export const TARGET_KINDS = ['dataset', 'task', 'family', 'sft_row', 'chain', 'teacher_demo', 'episode', 'call', 'corpus', 'run', 'held_task', 'held_episode', 'other'];
export const EDIT_ACTIONS = ['flag', 'unflag', 'exclude', 'include', 'annotate', 'correct'];
// Held-out data is view-only (AGENTS.md held-data separation): no command can record an edit against it.
export const PROTECTED_KINDS = ['held_task', 'held_episode'];

const target = { type: 'object', properties: {
  kind: { type: 'string', enum: TARGET_KINDS },
  dataset: str(256, 1),        // catalog dataset id, e.g. rl-task-supply/natlang-rl-task-supply-20261120c
  id: str(512, 1),             // the item's natural id inside the dataset (task id, row index, episode id, call key)
  ref: str(1024),              // locator the viewer used (path + offset / line), for provenance
  label: str(300),
}, required: ['kind', 'dataset', 'id'], additionalProperties: false };

const subject = { type: 'object', properties: { kind: str(40, 1), id: str(300, 1), title: str(300) },
  required: ['kind', 'id'], additionalProperties: false };

function readAll(view, topic) {
  const out = [];
  let cursor;
  for (;;) {
    const batch = view.readEvents({ topics: [topic], limit: 256, ...(cursor === undefined ? {} : { cursor }) });
    for (const e of batch.events) out.push({ ...e.payload, author: e.author, operationId: e.operationId });
    if (!batch.hasMore) return out;
    if (cursor === batch.cursor) throw Error('OPS_CURSOR_STALLED');
    cursor = batch.cursor;
  }
}

export const definition = defineApplication({
  id: 'natlang-ops', version: '1',
  commands: {
    proposeEdit: defineCommand({
      description: 'Record one reviewed edit of a training-data item: flag/unflag, exclude/include from training, annotate, or correct. '
        + 'The native participant applies it to the errata, exclusion or annotation layer and records the result.',
      input: { type: 'object', properties: {
        target, action: { type: 'string', enum: EDIT_ACTIONS },
        text: str(4000),                       // reason or note
        // corrected content (action correct), text or JSON, split into <= 16 KiB pieces (the schema bounds one string at 16 KiB)
        correction: { type: 'array', items: str(16384), maxItems: 16 },
        correctionFormat: { type: 'string', enum: ['text', 'json'] },
        field: str(200),                       // which field the correction replaces (e.g. oracle, answer, calls.3.completion)
        basedOn: str(128),                     // content digest of the item version the person reviewed
        supersedes: str(128),                  // operation id of an earlier edit this one replaces
      }, required: ['target', 'action', 'text'], additionalProperties: false },
      prepare(input) {
        if (PROTECTED_KINDS.includes(input.target.kind)) throw Error('HELD_DATA_IS_VIEW_ONLY');
        if (input.action === 'correct' && !(input.correction || []).join('')) throw Error('CORRECTION_REQUIRED');
        if (['exclude', 'flag', 'correct'].includes(input.action) && !input.text.trim()) throw Error('REASON_REQUIRED');
        return program(emit('edit', input));
      },
    }),
    recordApplication: defineCommand({
      description: 'Native participant: the outcome of applying one edit to the data layers (only native-authored records count).',
      input: { type: 'object', properties: {
        edit: str(128, 1), status: { type: 'string', enum: ['applied', 'refused', 'failed', 'superseded'] },
        detail: str(2000), files: str(2000),
      }, required: ['edit', 'status', 'detail'], additionalProperties: false },
      prepare: input => program(emit('edit.applied', input)),
    }),
    ask: defineCommand({
      description: 'Ask the supervising agent about an item, request a unit stop, or acknowledge an alert.',
      input: { type: 'object', properties: {
        kind: { type: 'string', enum: ['ask', 'request_stop', 'ack_alert'] }, subject, text: str(2000, 1),
      }, required: ['kind', 'subject', 'text'], additionalProperties: false },
      prepare: input => program(emit('ask', input)),
    }),
    answer: defineCommand({
      description: 'The supervising agent answers one question asked from the page.',
      input: { type: 'object', properties: { ask: str(128, 1), text: str(4000, 1) }, required: ['ask', 'text'], additionalProperties: false },
      prepare: input => program(emit('answer', input)),
    }),
  },
  queries: {
    edits: defineQuery({
      description: 'Every recorded edit with every application record, from one coherent view.',
      input: { type: 'null' },
      read: view => ({ edits: readAll(view, 'edit'), applications: readAll(view, 'edit.applied') }),
    }),
    inbox: defineQuery({
      description: 'Questions asked from the page and the supervising agent\'s answers.',
      input: { type: 'null' },
      read: view => ({ asks: readAll(view, 'ask'), answers: readAll(view, 'answer') }),
    }),
  },
});
