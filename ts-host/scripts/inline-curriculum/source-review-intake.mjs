/**
 * Source reviewers at intake (plans/SOURCE_REVIEW_PROGRAM.md, item P2).
 *
 * The intake readers (folder-data.mjs, workflow-sources.mjs, sources-ai2.mjs) note each item they keep. A build script then
 * asks `reviewSourceItem` about the noted items and writes the recommendations beside its output, as the review-output
 * lines that `python3 scripts/source_review.py receipt` turns into receipts. The reviewers are advice: no reader
 * consults the answer, so admission, ordering and every output byte stay as the crisp registry decides.
 *
 * The mode is `crisp` (default: nothing is noted, no model is called, no file is written), `nl` or `shadow`. Choose it with
 * `--source-review-mode` or NATLANG_SOURCE_REVIEW_MODE. `nl` and `shadow` need `--source-review-server` and
 * `--source-review-model` (an OpenAI-compatible endpoint), or an executor passed to `writeIntakeReviews`.
 */
import { appendFile, mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { pluggableMode } from '../../dist/runtime/pluggable.js';

/** The parseArgs options of the flags; spread them into a script's own options. */
export const SOURCE_REVIEW_OPTIONS = {
  'source-review-mode': { type: 'string' }, 'source-review-server': { type: 'string' },
  'source-review-model': { type: 'string' }, 'source-review-limit': { type: 'string' },
};

const MODES = ['crisp', 'nl', 'shadow'];
const state = { mode: 'crisp', limit: Infinity, server: undefined, model: undefined, items: new Map() };

/** Read the mode, endpoint and limit from a script's parsed `values` (flags win over the environment). */
export function configureSourceReview(values = {}, env = process.env) {
  const mode = values['source-review-mode'] ?? env.NATLANG_SOURCE_REVIEW_MODE ?? 'crisp';
  if (!MODES.includes(mode)) throw new Error(`--source-review-mode is one of ${MODES.join(', ')}; got ${JSON.stringify(mode)}`);
  const limit = values['source-review-limit'] === undefined ? Infinity : Number(values['source-review-limit']);
  if (!(limit > 0)) throw new Error('--source-review-limit is a positive number of items');
  Object.assign(state, { mode: pluggableMode(mode, 'crisp'), limit, items: new Map(),
    server: values['source-review-server'] ?? env.NATLANG_SOURCE_REVIEW_SERVER, model: values['source-review-model'] ?? env.NATLANG_SOURCE_REVIEW_MODEL });
  return state.mode;
}

export const sourceReviewMode = () => state.mode;

/** Note one kept item `{ dataset, id, visible, annotated_label, contract, answer_format }`. A no-op in `crisp` mode. */
export function noteSourceItem(item) {
  if (state.mode === 'crisp') return;
  const key = `${item.dataset}\0${item.id}`;
  if (!state.items.has(key) && state.items.size < state.limit) state.items.set(key, { answer_format: null, ...item });
}

export const notedSourceItems = () => [...state.items.values()];

async function defaultExecutor() {
  if (!state.server || !state.model) throw new Error('--source-review-server and --source-review-model are required for --source-review-mode nl or shadow');
  const { createNatlangRuntime, openAICompatibleModelTurn } = await import('../../dist/index.js');
  const { builtin } = await import('../../dist/runtime/builtin.js');
  const runtime = createNatlangRuntime({ model: openAICompatibleModelTurn({ endpoint: state.server, model: state.model }), seed: { mode: 'backend' }, calls: false });
  return { executor: state.model, close: () => runtime.close?.(),
    call: (name, args) => runtime.run(() => builtin(name)(...args)) };
}

/**
 * Review the noted items and write the recommendations to `path` (review-output lines, one per item). Returns null in
 * `crisp` mode without touching the file system. `executor` ({ executor, call }) replaces the endpoint, for tests.
 */
export async function writeIntakeReviews(path, { executor } = {}) {
  if (state.mode === 'crisp') return null;
  const { reviewSourceItem } = await import('../../dist/teacher/source-review-nl.js');
  const { pendingSourceReview } = await import('../../dist/teacher/source-review.js');
  const own = executor ? null : await defaultExecutor();
  const primary = executor ?? own;
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, '');
  const counts = { reviewed: 0, hold: 0, admit: 0, errors: 0, standing: 0 };
  try {
    for (const item of state.items.values()) {
      // An item with a registry entry has its standing decision; the reviewer is asked about new items only.
      if (pendingSourceReview(item.dataset, item.id)) { counts.standing++; continue; }
      let line;
      try {
        const output = await reviewSourceItem(item, { mode: state.mode, primary });
        if (!output) continue;
        line = output; counts.reviewed++; counts[output.recommendation.recommendation]++;
      } catch (error) {
        counts.errors++;
        line = { kind: 'item', input: { item, precedents: [] }, reviewer: { kind: 'crisp', function: 'reviewSourceItem' }, recommendation: null,
          excluded: 'transport_error', error: String(error?.message ?? error) };
      }
      await appendFile(path, JSON.stringify({ ...line, training_admission: false }) + '\n');
    }
  } finally { own?.close?.(); }
  return { path, mode: state.mode, noted: state.items.size, ...counts };
}
