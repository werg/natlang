/**
 * pi on natlang: pi-durable's harness with its built-in task kinds (`pi.generation`, `pi.tool`, `pi.compaction`)
 * carried out by natural-language functions, and pi's coding agent as extensions.
 *
 * The host is pi-durable itself, vendored at f10993b (`vendor/durable`): the Session line and SQLite storage, the
 * scheduler's mechanics, the registry and the Harness API. The registry the Harness reads resolves the built-in task
 * names to the natural-language kinds, so every task pi-durable creates (a run's generation, a round's tool tasks,
 * compactions) runs here. Context building, the system-entry plan and estimate, the scheduler's policy and admission are
 * pluggable (`pluggable()`, modes `crisp`, `nl`, `shadow`): pi's crisp code by default, the natural-language functions
 * when selected, or both compared in shadow mode. See PORT.md.
 */
import type { Context } from '@earendil-works/chord';
import { BACKGROUND_CONTEXT } from '@earendil-works/chord/context';
import type { Models } from '@earendil-works/pi-ai';
import type { NatlangRuntime } from '@natlang/node';
import { Harness, type HarnessOptions, type HarnessSettings } from './vendor/durable/src/index.ts';
import type { Extension, Registry } from './vendor/durable/src/harness/types.ts';
import type { Storage } from './vendor/durable/src/types.ts';
import { portOptions, type Implementations } from './host/harness.ts';
import { substituteTasks } from './host/registry.ts';
import type { Entry, TaskHost } from './host/tasks.ts';
import generation from './generation.nl';
import tool from './tool.nl';
import compaction from './compaction.nl';

export type PiOptions = {
  storage: Storage;
  /** Runs the natural-language functions (the executor). */
  natlang: NatlangRuntime;
  /** pi-ai model access: the agent's provider models. */
  models: Models;
  registry: Registry;
  settings?: HarnessSettings;
  env?: HarnessOptions['env'];
  now?: () => number;
  onReport?: (error: unknown) => void;
  onPhase?: TaskHost['onPhase'];
  /** Pluggable hot paths: `crisp` (the default), `nl` or `shadow` per point. */
  implementations?: Partial<Implementations>;
  /** Runs per phase before the task faults (default 2). */
  attempts?: number;
};

/** Open pi on `options.storage`: a Harness whose built-in task kinds are natural-language functions. */
export async function openPi(options: PiOptions, context: Context = BACKGROUND_CONTEXT): Promise<Harness> {
  const port = portOptions({
    models: options.models,
    registry: options.registry,
    ...(options.settings ? { settings: options.settings } : {}),
    ...(options.env ? { env: options.env } : {}),
    ...(options.now ? { now: options.now } : {}),
    ...(options.onReport ? { onReport: options.onReport } : {}),
  } as HarnessOptions, {
    natlang: options.natlang,
    entries: { generation: generation as Entry, tool: tool as Entry, compaction: compaction as Entry },
    implementations: options.implementations,
    attempts: options.attempts,
    onPhase: options.onPhase,
    now: options.now,
  });
  const harness = await Harness.open(options.storage, port.options, context);
  port.bind(harness);
  return harness;
}

export type { Extension, Harness, Implementations };
export { substituteTasks };
