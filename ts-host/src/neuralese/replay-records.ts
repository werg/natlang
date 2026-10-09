/**
 * Replay records (`natlang.replay-record/1`; plans/neuralese/S5_PROGRAM_TRAINING.md §3): every gradient session a
 * program runs (`valueAndGrad`, `grad`), kept as training data for whole-program replay
 * (training/neuralese/natlang_neuralese/train/graph_replay.py). A record is the session's request (argument blocks,
 * loss terms over recorded turns, and the producers: every recorded turn that wrote blocks) plus the loss the server
 * returned and the caller's annotations. Every block the record mentions is saved beside it, in the server's one-block
 * wire format, so the record replays without the server that ran it.
 *
 * Node only (it writes files). Pass the sink as `learningService({ ..., replayRecords })`.
 */
import { fetchModel } from '../model/chat-completion.js';

type Json = Record<string, unknown>;

export const REPLAY_RECORD_SCHEMA = 'natlang.replay-record/1';

/** What `evaluate` hands to a sink. */
export type ReplaySession = { arguments: string[]; terms: readonly Json[]; producers?: readonly Json[]; order: number;
  derived?: readonly Json[]; loss: number };

export interface ReplayRecordSink { record(session: ReplaySession): Promise<void> }

const BLOCK_ID = /nz1_[a-z2-7]+/g;

/**
 * A sink writing `<path>` (JSONL) and `<path without .jsonl>.blocks/<id>.safetensors`. `annotate` returns fields
 * merged into each record (operator, family, split, group, case, expected); `id` defaults to `<file stem>-<n>`.
 */
export function replayRecordSink(options: { path: string; endpoint: string; headers?: Record<string, string>;
  annotate?: () => Json }): ReplayRecordSink & { readonly count: number } {
  let count = 0;
  const base = options.endpoint.replace(/\/$/, '');
  return {
    get count() { return count; },
    async record(session) {
      const { default: fs } = await import('node:fs');
      const { default: path } = await import('node:path');
      const stem = options.path.replace(/\.jsonl$/, '');
      const blocks = `${stem}.blocks`;
      fs.mkdirSync(blocks, { recursive: true });
      const annotations = options.annotate?.() ?? {};
      const record = { schema: REPLAY_RECORD_SCHEMA, id: `${path.basename(stem)}-${count}`, ...annotations,
        arguments: session.arguments, terms: session.terms, ...(session.producers?.length ? { producers: session.producers } : {}),
        order: session.order, ...(session.derived?.length ? { derived: session.derived } : {}), loss: session.loss,
        recorded_at: new Date().toISOString() };
      const text = JSON.stringify(record);
      for (const id of new Set(text.match(BLOCK_ID) ?? [])) {
        const file = path.join(blocks, `${id}.safetensors`);
        if (fs.existsSync(file)) continue;
        const response = await fetchModel(`${base}/v1/neuralese/blocks/${id}`, { headers: options.headers });
        if (!response.ok) throw new Error(`replay record: block ${id} is not on the server (${response.status})`);
        fs.writeFileSync(file, new Uint8Array(await response.arrayBuffer()));
      }
      fs.appendFileSync(options.path, text + '\n');
      count++;
    },
  };
}
