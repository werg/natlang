/**
 * Trained soft forms of the runtime's prompt pieces, read from a `natlang_neuralese.tune.build` result
 * (docs/TUNING.md "Soft states"; plans/neuralese/DECISIONS.md 2026-11-20 "one primed-state family").
 *
 * Every soft prompt is one kind of thing: a native state primed from its text in its real rendered call
 * (`tune/soft.py` `initialize_soft_states`), then trained on verified rows. A program binding's instruction state, a
 * memetic guidance state and a learned operator's prompt state differ only in data: the binding they belong to. A
 * runtime prompt piece's state belongs to the binding `prompt:<piece>` (every site of the piece) or
 * `prompt:<piece>@<scope>` (one site: `view@compact-value` is the compact-value operator's own state of view's body;
 * system-prompts.ts `scopedBank`). The build's `result.json` lists them as `softStates` and ships each as its store file
 * `states/<id>.nzs`, the native-state format every engine reads.
 *
 * Node only (reads the build directory).
 */
import type { NeuraleseStore } from '../native/neuralese-store.js';
import { neuraleseRef } from '../native/neuralese.js';
import { hexDigest } from '../native/hash.js';
import { PIECE_SCOPE, promptPieces, SYSTEM_PROMPT_TYPE, systemPromptBank, type PromptPiece, type SystemPromptBank } from '../native/system-prompts.js';

/** The binding namespace of runtime prompt pieces' soft states. */
export const PROMPT_BINDING = 'prompt:';

export type PromptSoftState = { binding: string; kind: string; target?: string | null; state: string; textSha256: string };

/** The bank entries `{piece[@scope]: state}` of a build's soft states; refuses a state primed from another text than
 * this runtime's piece (a stale state stands for text the runtime no longer sends). */
export function promptStateIds(softStates: readonly PromptSoftState[], pieces: readonly PromptPiece[] = promptPieces()): Record<string, string> {
  const texts = new Map(pieces.map(piece => [piece.id, piece.text]));
  const out: Record<string, string> = {};
  for (const item of softStates) {
    if (!item.binding.startsWith(PROMPT_BINDING) || item.kind !== 'instructions') continue;
    const id = item.binding.slice(PROMPT_BINDING.length);
    const at = id.lastIndexOf(PIECE_SCOPE);
    const text = texts.get(at > 0 ? id.slice(0, at) : id);
    if (text === undefined) throw new Error(`system-prompt-unknown-piece: ${id} is not a prompt piece of this runtime`);
    if (item.textSha256 !== hexDigest(text))
      throw new Error(`system-prompt-stale: the soft state of ${id} was primed from another text than this runtime's piece`);
    out[id] = item.state;
  }
  return out;
}

/** Load a build's prompt-piece soft states into `store` (their wires as stored) and return them as a bank. */
export async function loadPromptStates(directory: string, store: NeuraleseStore): Promise<SystemPromptBank> {
  const { readFileSync } = await import('node:fs');
  const { join } = await import('node:path');
  const result = JSON.parse(readFileSync(join(directory, 'result.json'), 'utf8')) as {
    softStates?: PromptSoftState[]; softStateFiles?: Record<string, string> };
  const ids = promptStateIds(result.softStates ?? []);
  if (!Object.keys(ids).length) throw new Error(`system-prompt-states-empty: ${directory} holds no prompt-piece soft states`);
  if (!store.putNativeStateWire) throw new Error('neuralese-native-archive-unavailable: prompt states need a native-capable store');
  for (const state of new Set(Object.values(ids))) {
    const file = result.softStateFiles?.[state] ?? `states/${state}.nzs`;
    const saved = await store.putNativeStateWire(new Uint8Array(readFileSync(join(directory, file))));
    if (saved.id !== state) throw new Error(`neuralese-block-integrity: ${file} does not hold state ${state}`);
  }
  return systemPromptBank(Object.fromEntries(Object.entries(ids).map(([id, state]) => [id, neuraleseRef(SYSTEM_PROMPT_TYPE, state)])));
}

/** The prompt-state build a server-backed runtime loads: `NATLANG_NEURALESE_PROMPT_STATES` (a build directory), unset
 * or `off` for none. */
export function promptStatesDirectory(): string | undefined {
  const value = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.NATLANG_NEURALESE_PROMPT_STATES;
  return value && value !== 'off' ? value : undefined;
}
