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
import { HttpNeuraleseStore } from '../model/neuralese-server.js';
import { MemoryNeuraleseStore, archiveNativeArtifactFrom, nativeArtifactDependencyClosure,
  encodeNativeState, encodeNativeTensorArtifact } from '../native/neuralese-store.js';

type Json = Record<string, unknown>;

export const REPLAY_RECORD_SCHEMA = 'natlang.replay-record/1';

/** What `evaluate` hands to a sink. */
export type ReplaySession = { arguments: string[]; terms: readonly Json[]; producers?: readonly Json[]; order: number;
  derived?: readonly Json[]; native_turns?: readonly Json[]; native_graph_nodes?: readonly Json[];
  native_gradient?: string; native_gradients?: Record<string, string>; native_targets?: readonly Json[];
  derivative_execution_id?: string;
  target?: Json; loss: number };

export interface ReplayRecordSink { record(session: ReplaySession): Promise<void> }

const ARTIFACT_ID = /(?:nzs1|nnt1)_[a-z2-7]{52}/g;

/**
 * A sink writing `<path>` (JSONL) and `<path without .jsonl>.native/<id>.bin`. `annotate` returns fields
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
      const native = `${stem}.native`;
      fs.mkdirSync(native, { recursive: true });
      const annotations = options.annotate?.() ?? {};
      const record = { schema: REPLAY_RECORD_SCHEMA, id: `${path.basename(stem)}-${count}`, ...annotations,
        arguments: session.arguments, terms: session.terms, ...(session.producers?.length ? { producers: session.producers } : {}),
        order: session.order, ...(session.derived?.length ? { derived: session.derived } : {}),
        ...(session.native_turns?.length ? { native_turns: session.native_turns } : {}),
        ...(session.native_graph_nodes?.length ? { native_graph_nodes: session.native_graph_nodes } : {}),
        ...(session.native_gradient ? { native_gradient: session.native_gradient } : {}), ...(session.target ? { target: session.target } : {}), loss: session.loss,
        recorded_at: new Date().toISOString() };
      const text = JSON.stringify(record);
      const remote = new HttpNeuraleseStore(base, options.headers), archive = new MemoryNeuraleseStore();
      const roots = new Set(text.match(ARTIFACT_ID) ?? []);
      for (const id of roots) await archiveNativeArtifactFrom(remote, archive, id);
      const closure = await nativeArtifactDependencyClosure(archive, roots);
      for (const id of closure) {
        const file = path.join(native, `${id}.bin`);
        if (fs.existsSync(file)) continue;
        const state = await archive.getNativeState(id);
        const tensor = state ? undefined : await archive.getNativeTensorArtifact(id);
        if (!state && !tensor) throw new Error(`replay record: native dependency ${id} was not archived`);
        fs.writeFileSync(file, state ? encodeNativeState(state) : encodeNativeTensorArtifact(tensor!));
      }
      fs.appendFileSync(options.path, text + '\n');
      count++;
    },
  };
}
