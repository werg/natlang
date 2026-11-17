/**
 * Hermetic episodes. A collector episode (training, held evaluation, replay, teacher generation) runs with a fresh call
 * store of its own (machine-store.ts createEpisodeCallStore): what earlier episodes or runs stored (compiled cases, tier
 * outputs, iterateOn statistics, refinement verdicts) can never change what this one does, while its own records stay
 * readable within it (the item-result binding reads its host captures). Every runtime first used inside
 * `runInEpisodeCallStore`, worker threads started with `episodeCallStoreRoot()` in their data included, records to it
 * instead of the machine's store. Its settings switch specialization off, so nothing compiled within the episode serves
 * its later calls either. This scope lives in runtime/ so the evaluation and improvement workers that join it need no
 * import of the root layer.
 */
import { AsyncLocalStorage } from 'node:async_hooks';

const episodeStore = new AsyncLocalStorage<string>();
/** The episode store's root when the caller runs inside one. */
export function episodeCallStoreRoot(): string | undefined { return episodeStore.getStore(); }
/** Run `body` with the store at `root` as the default store of every runtime it creates. */
export function runInEpisodeCallStore<T>(root: string, body: () => T): T { return episodeStore.run(root, body); }
/** A worker thread joins its parent's episode store for the rest of its life. */
export function enterEpisodeCallStore(root: string | undefined): void { if (root) episodeStore.enterWith(root); }
