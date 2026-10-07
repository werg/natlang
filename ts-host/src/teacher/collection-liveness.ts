/** Minimal, non-retaining diagnostics for a collector promise that never settles. */
export type CollectionLivenessSnapshot = {
  stage: string;
  active_cases?: Array<{ index: number; program_id: string }>;
};

function boundedSnapshot(snapshot: CollectionLivenessSnapshot): Record<string, unknown> {
  const active = (snapshot.active_cases ?? []).slice(0, 8).map(item => ({
    index: Number.isSafeInteger(item.index) ? item.index : null,
    program_id: String(item.program_id).slice(0, 160),
  }));
  return {
    stage: String(snapshot.stage).slice(0, 80), active_cases: active,
    active_cases_truncated: (snapshot.active_cases?.length ?? 0) > active.length,
    pending_calls: { available: false }, provider_state: { available: false },
  };
}

/** Detect Node's beforeExit edge while a real collection promise remains pending. */
export async function observeCollectionPromise<T>(
  run: Promise<T>, snapshot: () => CollectionLivenessSnapshot,
): Promise<T> {
  let pending = true;
  let reported = false;
  const onBeforeExit = () => {
    if (!pending || reported) return;
    reported = true;
    process.stderr.write(JSON.stringify({ event: 'incomplete_collection',
      version: 'natlang.incomplete_collection/1', ...boundedSnapshot(snapshot()) }) + '\n');
    process.exitCode ||= 1;
  };
  process.on('beforeExit', onBeforeExit);
  try { return await run; }
  finally {
    pending = false;
    process.off('beforeExit', onBeforeExit);
  }
}
