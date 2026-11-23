/**
 * The reply a decision readout scores, as pure functions with no import (staged with the Pi target): the runtime asks for
 * each allowed value as one JSON value (`decisionPrompt`), and the scorer of a model that does not answer in that form
 * asks for and scores the form it does (`framedDecision`).
 *
 *   - `value` (default): the JSON value itself, `"yes"`. The decision trainer's rows are written this way
 *     (training/neuralese/natlang_neuralese/data/decision_graph.py), so a model natlang trained answers in it.
 *   - `result-object`: `{"result": "yes"}`. An instruction model that natlang did not train does not answer a function
 *     call with a bare JSON string. After the closed reasoning of its template it opens a tool call or an object with the
 *     key `result` (LFM3-1.7B, greedy: `{"result": "refusal"}`). Scoring `"yes"` there evaluates a reply it gives about
 *     e^-15 of the time at its first token, and what separates the options at that point is not the evidence but where
 *     each is listed and how its first token is merged (plans/MODEL_FACING_CHANGES.md, 2026-11-12). Scoring the form
 *     the model writes puts the choice at the token where the options differ.
 */
export type DecisionReply = 'value' | 'result-object';
export const DECISION_REPLIES: readonly DecisionReply[] = ['value', 'result-object'];
/**
 * A profile's `decisionReply`: one form, or `auto` (the default except for a managed stock model): a readout without
 * reasoning is read in the form the model puts more probability on (`autoFramedScores`). Whether a checkpoint answers a
 * readout with the bare value right after its empty think block depends on how it was trained, which its files do not
 * say: the B14 triage bench's native checkpoint 5ff07638 (the checkpoint a stale `lfm-latest` profile served) puts
 * ~e^-8 on the bare value there and opens a tool call instead, so its option scores followed the listing order and the
 * value's first-token merges (`billing` for spam, `true` on every ticket: 4/28, against 26/28 in `{"result": VALUE}`),
 * while aei1's collection policy e15b20a8 puts nearly all of its mass on the bare value (DECISIONS 2026-11-23
 * "instant readout collapse").
 */
export type DecisionReplySetting = DecisionReply | 'auto';
export const DECISION_REPLY_SETTINGS: readonly DecisionReplySetting[] = [...DECISION_REPLIES, 'auto'];

/** The prompt that asks for one of `replies` (JSON texts) in `frame`. */
export const decisionPrompt = (replies: readonly string[], frame: DecisionReply = 'value') => frame === 'result-object' ?
  'Answer this call now, without tools. Reply with exactly one JSON object {"result": VALUE} and nothing else, where VALUE is one of ' +
  'these JSON values: ' + replies.join(', ') + '.' :
  'Answer this call now, without tools. Reply with exactly one of these JSON values and nothing else: ' + replies.join(', ') + '.';

/** The reply text of one value (its JSON) in `frame`; `open` is the start of a reply (a field-by-field step), not closed. */
export const decisionReply = (json: string, frame: DecisionReply = 'value', open = false) =>
  frame === 'result-object' ? `{"result": ${json}${open ? '' : '}'}` : json;

const FIELDS_LEAD = 'Answer this call now, without tools. Reply with exactly one JSON value of ';
const FIELDS_ONE = 'this form and nothing else: ', FIELDS_MANY = 'one of these forms and nothing else: ';
const FIELDS_FRAMED = 'Answer this call now, without tools. Reply with exactly one JSON object {"result": VALUE} and nothing else, where VALUE is a JSON value of ';

/**
 * The prompt of a decision over a record type too large to list (native/decision.ts `decisionDomain`): each form is
 * one arm of the type as compact JSON with every field's values written `<A | B>` (`formText`), and the reply is
 * scored field by field.
 */
export const decisionFieldsPrompt = (forms: readonly string[], frame: DecisionReply = 'value') =>
  (frame === 'result-object' ? FIELDS_FRAMED : FIELDS_LEAD) + (forms.length === 1 ? FIELDS_ONE : FIELDS_MANY) + forms.join(', ') + '.';

/** The forms of a `decisionFieldsPrompt` (value frame), or undefined when `content` is not one. */
function fieldsForms(content: string): string | undefined {
  if (!content.startsWith(FIELDS_LEAD) || !content.endsWith('.')) return undefined;
  const rest = content.slice(FIELDS_LEAD.length);
  return rest.startsWith(FIELDS_ONE) ? rest.slice(FIELDS_ONE.length) : rest.startsWith(FIELDS_MANY) ? rest.slice(FIELDS_MANY.length) : undefined;
}

/**
 * A readout request (the runtime's: its last message is `decisionPrompt(options)` and its options are JSON values) as
 * the model of `frame` is asked and scored. A request that is not a readout (a judge's own instructions) is returned
 * as it is. Scores come back in the options' order.
 */
export function framedDecision<T extends { messages: unknown[]; options: string[]; open_reply?: boolean }>(request: T, frame: DecisionReply | undefined): T {
  if (frame === undefined || frame === 'value') return request;
  const last = request.messages.at(-1) as { role?: unknown; content?: unknown } | undefined;
  if (request.open_reply) {
    const forms = last?.role === 'user' && typeof last.content === 'string' ? fieldsForms(last.content) : undefined;
    if (forms === undefined) return request;
    const lead = last!.content as string;
    return { ...request, messages: [...request.messages.slice(0, -1), { ...last, content: FIELDS_FRAMED + lead.slice(FIELDS_LEAD.length) }],
      options: request.options.map(option => decisionReply(option, frame, true)) };
  }
  if (last?.role !== 'user' || last.content !== decisionPrompt(request.options)) return request;
  return { ...request, messages: [...request.messages.slice(0, -1), { ...last, content: decisionPrompt(request.options, frame) }],
    options: request.options.map(option => decisionReply(option, frame)) };
}

/** The `decisionReply` in force: `NATLANG_DECISION_REPLY` (value, result-object or auto) overrides the profile's. */
export function decisionReplySetting(configured: DecisionReplySetting | undefined): DecisionReplySetting | undefined {
  const env = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.NATLANG_DECISION_REPLY;
  if (env === undefined || env === '') return configured;
  if (!(DECISION_REPLY_SETTINGS as readonly string[]).includes(env))
    throw new Error(`NATLANG_DECISION_REPLY must be ${DECISION_REPLY_SETTINGS.join(', ')}, not ${env}`);
  return env as DecisionReplySetting;
}

const logSumExp = (values: readonly number[]) => {
  const top = Math.max(...values);
  return top === -Infinity ? top : top + Math.log(values.reduce((sum, value) => sum + Math.exp(value - top), 0));
};

/**
 * Score a readout in the reply form `reply` names (`framedDecision`); under `auto`, in the form the model answers in.
 *
 * `auto` applies to a joint readout without reasoning (after its own reasoning the model answers in the form the
 * request asks for, the bare value; a field-by-field step continues the form its first step chose, the value): the
 * request is scored in both forms, and each option of each form is also scored alone. An option scored alone keeps
 * the tokens it shares with the others (`</think>`, the object's opening), so the sum of a form's lone scores is the
 * model's probability of answering in that form at all. The form with more of it is read: its scores (the options
 * together, the shared tokens cancelled exactly) are returned with `reply_frame` and both forms' `reply_masses`
 * (natural log). One round of 2 + 2K items, which a batching server scores together.
 */
export async function autoFramedScores<T extends { messages: unknown[]; options: string[]; open_reply?: boolean; reasoning?: string },
  S extends { log_probs: number[] }>(request: T, reply: DecisionReplySetting | undefined, scoreAll: (items: T[]) => Promise<S[]>):
    Promise<S & { reply_frame?: DecisionReply; reply_masses?: Record<DecisionReply, number> }> {
  const last = request.messages.at(-1) as { role?: unknown; content?: unknown } | undefined;
  const readout = last?.role === 'user' && last.content === decisionPrompt(request.options);
  if (reply !== 'auto' || !readout || request.open_reply || typeof request.reasoning === 'string' || request.options.length < 2)
    return (await scoreAll([framedDecision(request, reply === 'auto' ? 'value' : reply)]))[0]!;
  const frames = DECISION_REPLIES.map(frame => framedDecision(request, frame));
  const alone = frames.flatMap(item => item.options.map(option => ({ ...item, options: [option] })));
  const scored = await scoreAll([...frames, ...alone]);
  const k = request.options.length;
  const masses = DECISION_REPLIES.map((_, index) => logSumExp(scored.slice(frames.length + index * k, frames.length + (index + 1) * k)
    .map(scores => scores.log_probs[0]!)));
  const pick = masses[1]! > masses[0]! ? 1 : 0;
  return { ...scored[pick]!, reply_frame: DECISION_REPLIES[pick]!,
    reply_masses: Object.fromEntries(DECISION_REPLIES.map((frame, index) => [frame, masses[index]!])) as Record<DecisionReply, number> };
}
