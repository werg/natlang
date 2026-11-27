// natlang ops: the retained part of the application (Chronolog v2). One definition shared by the agent program on the DGX
// (agent.mjs) and the browser Worker (worker.mjs). Reducers are hash-pinned ES Modules in `$reducers` and run in QuickJS on every
// replica; there is no separate domain bundle.
//
// Retained facts are small and human-made:
//   - training-data edits (flag/exclude/annotate/correct) and the agent program's record of applying each one to the
//     errata/exclusion/annotation layers (`ops/edit`, `ops/applied` event topics);
//   - questions to and answers from the supervising agent: the starter's agent-request pipe (`agent/message`,
//     `agent/receipt`; shared/helpers/agent-requests.mjs). A question is a chat message whose `data` is {kind, subject};
//     an answer is a message with `inReplyTo`; a receipt says the inbox worker has it.
// Live ops state and training data are NOT retained here: the agent program serves them as live data feeds over session signals
// (feeds.mjs), so the signed journal never grows with 30 s snapshots or GB-scale rollouts.
// Changing these semantics means a new definition version and `npm run init` for a new page (the genesis pins the reducers).
import { defineApplication, defineCommand, defineQuery } from '@chronolog/peer-session';
import { agentRequestCommands, agentRequestQueries, agentRequestTopics } from '../../shared/helpers/agent-requests.mjs';

const str = (max, min = 0) => ({ type: 'string', minLength: min, maxLength: max });
export const TARGET_KINDS = ['dataset', 'task', 'family', 'sft_row', 'chain', 'teacher_demo', 'episode', 'call', 'corpus', 'run', 'held_task', 'held_episode', 'other'];
export const EDIT_ACTIONS = ['flag', 'unflag', 'exclude', 'include', 'annotate', 'correct'];
// Held-out data is view-only (AGENTS.md held-data separation): no command can record an edit against it.
export const PROTECTED_KINDS = ['held_task', 'held_episode'];
export const TOPICS = Object.freeze({ edit: 'ops/edit', applied: 'ops/applied' });
/** The kinds of question the page sends to the supervising agent (the `kind` inside an agent message's `data`). */
export const ASK_KINDS = ['ask', 'request_stop', 'ack_alert'];

const target = { type: 'object', properties: {
  kind: { type: 'string', enum: TARGET_KINDS },
  dataset: str(256, 1),        // catalog dataset id, e.g. rl-task-supply/natlang-rl-task-supply-20261120c
  id: str(512, 1),             // the item's natural id inside the dataset (task id, row index, episode id, call key)
  ref: str(1024),              // locator the viewer used (path + offset / line), for provenance
  label: str(300),
}, required: ['kind', 'dataset', 'id'], additionalProperties: false };

/** One reducer for every event-log topic: a writer's input is retained as an event on `config.eventTopic` and read back with
 * `readEvents`. State is only a counter, so replicas stay small however long the log gets. */
export const LOG_REDUCER_SOURCE = `
export default {
  reduce(state, tx, ctx) {
    if (!ctx.query('$permissions', { check: 'writer', actor: tx.author })) return { ok: false, code: 'writer_required' };
    const topic = ctx.config && typeof ctx.config.eventTopic === 'string' ? ctx.config.eventTopic : 'ops/event';
    const count = (state && typeof state.count === 'number' ? state.count : 0) + 1;
    return { ok: true, state: { count, lastAuthor: tx.author }, events: [{ topic, payload: tx.payload }] };
  },
  query(state) { return state || { count: 0 }; },
};
`.trimStart();

export const imports = Object.freeze({ 'app/ops-log': LOG_REDUCER_SOURCE });
export const topics = Object.freeze(Object.fromEntries(
  [...Object.values(TOPICS), ...agentRequestTopics].map(name => [name,
    Object.freeze({ module: 'app/ops-log', config: Object.freeze({ eventTopic: name }) })])));

/** Every event of `names`, oldest first, from one coherent view. */
export async function readAll(view, names) {
  const out = [];
  let cursor;
  for (;;) {
    const batch = await view.readEvents({ topics: names, limit: 128, ...(cursor === undefined ? {} : { cursor }) });
    for (const e of batch.events) out.push({ topic: e.topic, ...e.payload, author: e.author, operationId: e.operationId, atMs: Number(e.authorTimestampMs) || null });
    if (!batch.hasMore) return out;
    if (cursor === batch.cursor) throw Error('OPS_CURSOR_STALLED');
    cursor = batch.cursor;
  }
}
export async function deriveEdits(view) {
  const all = await readAll(view, [TOPICS.edit, TOPICS.applied]);
  return { edits: all.filter(e => e.topic === TOPICS.edit), applications: all.filter(e => e.topic === TOPICS.applied) };
}

export const definition = defineApplication({
  id: 'natlang-ops', version: '2',
  commands: {
    proposeEdit: defineCommand({
      description: 'Record one reviewed edit of a training-data item: flag/unflag, exclude/include from training, annotate, or correct. '
        + 'The agent program applies it to the errata, exclusion or annotation layer and records the result.',
      input: { type: 'object', properties: {
        target, action: { type: 'string', enum: EDIT_ACTIONS },
        text: str(4000),                       // reason or note
        // corrected content (action correct), text or JSON, split into <= 16 KiB pieces (the schema bounds one string at 16 KiB)
        correction: { type: 'array', items: str(16384), maxItems: 16 },
        correctionFormat: { type: 'string', enum: ['text', 'json'] },
        field: str(200),                       // which field the correction replaces (e.g. oracle, answer, calls.3.completion)
        basedOn: str(128),                     // content digest of the item version the person reviewed
        supersedes: str(512),                  // operation id of an earlier edit this one replaces
      }, required: ['target', 'action', 'text'], additionalProperties: false },
      prepare(input) {
        if (PROTECTED_KINDS.includes(input.target.kind)) throw Error('HELD_DATA_IS_VIEW_ONLY');
        if (input.action === 'correct' && !(input.correction || []).join('')) throw Error('CORRECTION_REQUIRED');
        if (['exclude', 'flag', 'correct'].includes(input.action) && !input.text.trim()) throw Error('REASON_REQUIRED');
        return [{ topic: TOPICS.edit, input }];
      },
    }),
    recordApplication: defineCommand({
      description: 'Agent program: the outcome of applying one edit to the data layers (only records authored by the agent count).',
      input: { type: 'object', properties: {
        edit: str(512, 1), status: { type: 'string', enum: ['applied', 'refused', 'failed', 'superseded'] },
        detail: str(2000), files: str(2000),
      }, required: ['edit', 'status', 'detail'], additionalProperties: false },
      prepare: input => [{ topic: TOPICS.applied, input }],
    }),
    // Questions and answers ride the starter's agent-request pipe: agentMessage (the question is `text` plus `data` =
    // JSON {kind, subject}; the answer carries `inReplyTo`) and agentReceipt.
    ...agentRequestCommands,
  },
  queries: {
    edits: defineQuery({
      description: 'Every recorded edit with every application record, from one coherent view.',
      input: { type: 'null' },
      read: deriveEdits,
    }),
    ...agentRequestQueries,
  },
});
