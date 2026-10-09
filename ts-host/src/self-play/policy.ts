import { builtinSource } from '../builtin/index.js';
import { fingerprint } from '../adaptation/identity.js';
import { createNatlangRuntime, type ModelDriver } from '../runtime/node.js';
import { loadVirtualNatlang } from '../runtime/virtual-project.js';
import type { InvocationTrace } from '../runtime/node.js';
import type { GameDecision, GamePolicy } from './types.js';

/** The game policy's source: the built-in program builtin/gamePolicy.nl, played as `play.nl` in a match's files. */
export const GAME_POLICY_SOURCE = builtinSource('gamePolicy');
export type PolicySnapshot = { files: Record<string, string>; entry: string; model: string };
export type PolicyDecisionAudit = { view: GameDecision; traces: InvocationTrace[]; action?: unknown; error?: string };

/** A source snapshot is frozen for the match; newly edited skills affect later matches only. */
export function natlangGamePolicy(options: { snapshot: PolicySnapshot; driver: ModelDriver;
  seed: number; onDecision?: (audit: PolicyDecisionAudit) => void | Promise<void> }): GamePolicy {
  const snapshot = structuredClone(options.snapshot);
  const target = loadVirtualNatlang(snapshot.files, snapshot.entry);
  const id = fingerprint({ version: 'natlang.game-policy/1', snapshot, seed: options.seed,
    model: { maxTurns: 6, maxTokens: 8192, turnTokens: 2048, maxFailureRepairs: 2 } });
  return { id, async decide(view, signal) {
    const audit: PolicyDecisionAudit = { view: structuredClone(view), traces: [] };
    const runtime = createNatlangRuntime({ model: { driver: options.driver, maxTurns: 6,
      maxTokens: 8192, turnTokens: 2048, maxFailureRepairs: 2 }, network: false, codeEdits: 'deny', signal,
      seed: { mode: 'derived', root: options.seed + view.decision },
      trace: trace => audit.traces.push(structuredClone(trace)), limits: { maxEpisodes: 18 } });
    try {
      const value = await runtime.run(() => target(JSON.stringify(view)));
      if (typeof value !== 'string') throw Error('game policy did not return a JSON action string');
      try { audit.action = JSON.parse(value); }
      catch { throw Error('game policy returned invalid action JSON'); }
      return audit.action;
    } catch (error) { audit.error = String(error); throw error; }
    finally { runtime.close(); await options.onDecision?.(structuredClone(audit)); }
  } };
}
