/** A provider turn exceeded its local deadline. Whole-case collection must not retry this implicitly. */
export class ProviderRequestTimeoutError extends Error {
  readonly code = 'NATLANG_PROVIDER_REQUEST_TIMEOUT';
  constructor(readonly metadata: { role: 'teacher' | 'judge'; phase: 'provider_prepare' | 'provider_turn';
    requestOrdinal: number | null; timeoutMs: number; startedAt: string;
    deadlineAt: string }) {
    super(`${metadata.role} provider ${metadata.phase === 'provider_prepare' ? 'preparation' : 'turn'} exceeded ${metadata.timeoutMs}ms deadline`);
    this.name = 'ProviderRequestTimeoutError';
  }
}

export class ProviderActionCycleTimeoutError extends Error {
  readonly code = 'NATLANG_PROVIDER_ACTION_CYCLE_TIMEOUT';
  constructor(readonly metadata: { role: 'teacher' | 'judge'; timeoutMs: number; startedAt: string; deadlineAt: string }) {
    super(`${metadata.role} provider action cycle exceeded ${metadata.timeoutMs}ms deadline`);
    this.name = 'ProviderActionCycleTimeoutError';
  }
}

function report(event: Record<string, unknown>): void {
  process.stderr.write(`${JSON.stringify({ event: 'provider_request_phase', ...event })}\n`);
}

/**
 * Bound one provider request and relay collection cancellation to the provider SDK. The race makes the collector
 * return on time even if a provider adapter ignores AbortSignal; abort remains best-effort in that case.
 */
export async function withProviderRequestDeadline<T>(options: {
  role: 'teacher' | 'judge';
  provider: string;
  phase: 'provider_prepare' | 'provider_turn';
  requestOrdinal: number | null;
  timeoutMs?: number;
  parentSignal?: AbortSignal;
  call(signal: AbortSignal): Promise<T>;
}): Promise<T> {
  const { provider, requestOrdinal, timeoutMs, parentSignal } = options;
  if (requestOrdinal !== null && (!Number.isSafeInteger(requestOrdinal) || requestOrdinal < 1))
    throw new RangeError('requestOrdinal must be positive or null');
  if (timeoutMs !== undefined && (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 2_147_483_647))
    throw new RangeError('provider request timeout must be a positive safe integer no greater than 2147483647ms');
  if (parentSignal?.aborted) throw parentSignal.reason instanceof Error ? parentSignal.reason : new Error('collection cancelled');
  const started = Date.now(), startedAt = new Date(started).toISOString();
  const deadlineAt = timeoutMs === undefined ? undefined : new Date(started + timeoutMs).toISOString();
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let removeParentAbort = () => {};
  report({ phase: options.phase, status: 'started', role: options.role, provider, request_ordinal: requestOrdinal,
    started_at: startedAt, ...(deadlineAt ? { deadline_at: deadlineAt, timeout_ms: timeoutMs } : {}) });

  const timedOut = timeoutMs === undefined ? new Promise<never>(() => undefined) : new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      const error = new ProviderRequestTimeoutError({ role: options.role, phase: options.phase, requestOrdinal, timeoutMs,
        startedAt, deadlineAt: deadlineAt! });
      reject(error);
      controller.abort(error);
    }, timeoutMs);
  });
  const cancelled = new Promise<never>((_resolve, reject) => {
    const abort = () => {
      const reason = parentSignal?.reason;
      reject(reason instanceof Error ? reason : new Error('collection cancelled'));
      controller.abort(reason);
    };
    if (parentSignal?.aborted) abort();
    else if (parentSignal) {
      parentSignal.addEventListener('abort', abort, { once: true });
      removeParentAbort = () => parentSignal.removeEventListener('abort', abort);
    }
  });
  try {
    const result = await Promise.race([Promise.resolve().then(() => options.call(controller.signal)), timedOut, cancelled]);
    report({ phase: options.phase, status: 'completed', role: options.role, provider, request_ordinal: requestOrdinal,
      started_at: startedAt, ended_at: new Date().toISOString(), elapsed_ms: Date.now() - started });
    return result;
  } catch (error) {
    report({ phase: options.phase, status: error instanceof ProviderRequestTimeoutError ? 'deadline_exceeded' :
      error instanceof ProviderActionCycleTimeoutError ? 'action_cycle_deadline_exceeded' :
      parentSignal?.aborted ? 'collection_aborted' : 'failed', role: options.role, provider, request_ordinal: requestOrdinal,
      started_at: startedAt, ended_at: new Date().toISOString(), elapsed_ms: Date.now() - started,
      ...(error instanceof ProviderRequestTimeoutError ? { deadline_at: deadlineAt, timeout_ms: timeoutMs } : {}) });
    throw error;
  } finally {
    if (timer) clearTimeout(timer);
    removeParentAbort();
  }
}

/** Bound preparation and all provider turns in one logical teacher/judge action cycle. */
export async function withProviderActionCycle<T>(options: {
  role: 'teacher' | 'judge';
  provider: string;
  timeoutMs?: number;
  parentSignal?: AbortSignal;
  call(signal: AbortSignal): Promise<T>;
}): Promise<T> {
  const { role, provider, timeoutMs, parentSignal } = options;
  if (timeoutMs !== undefined && (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 2_147_483_647))
    throw new RangeError('provider action cycle timeout must be a positive safe integer no greater than 2147483647ms');
  if (parentSignal?.aborted) throw parentSignal.reason instanceof Error ? parentSignal.reason : new Error('collection cancelled');
  const started = Date.now(), startedAt = new Date(started).toISOString();
  const deadlineAt = timeoutMs === undefined ? undefined : new Date(started + timeoutMs).toISOString();
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let removeParentAbort = () => {};
  report({ phase: 'provider_action_cycle', status: 'started', role, provider,
    started_at: startedAt, ...(deadlineAt ? { deadline_at: deadlineAt, timeout_ms: timeoutMs } : {}) });
  const timedOut = timeoutMs === undefined ? new Promise<never>(() => undefined) : new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      const error = new ProviderActionCycleTimeoutError({ role, timeoutMs, startedAt, deadlineAt: deadlineAt! });
      reject(error);
      controller.abort(error);
    }, timeoutMs);
  });
  const cancelled = new Promise<never>((_resolve, reject) => {
    const abort = () => {
      const reason = parentSignal?.reason;
      reject(reason instanceof Error ? reason : new Error('collection cancelled'));
      controller.abort(reason);
    };
    if (parentSignal?.aborted) abort();
    else if (parentSignal) {
      parentSignal.addEventListener('abort', abort, { once: true });
      removeParentAbort = () => parentSignal.removeEventListener('abort', abort);
    }
  });
  try {
    const result = await Promise.race([Promise.resolve().then(() => options.call(controller.signal)), timedOut, cancelled]);
    report({ phase: 'provider_action_cycle', status: 'completed', role, provider,
      started_at: startedAt, ended_at: new Date().toISOString(), elapsed_ms: Date.now() - started });
    return result;
  } catch (error) {
    report({ phase: 'provider_action_cycle', status: error instanceof ProviderActionCycleTimeoutError ? 'deadline_exceeded' :
      error instanceof ProviderRequestTimeoutError ? 'provider_request_deadline_exceeded' :
      parentSignal?.aborted ? 'collection_aborted' : 'failed', role, provider, started_at: startedAt,
      ended_at: new Date().toISOString(), elapsed_ms: Date.now() - started,
      ...(error instanceof ProviderActionCycleTimeoutError ? { deadline_at: deadlineAt, timeout_ms: timeoutMs } : {}) });
    throw error;
  } finally {
    if (timer) clearTimeout(timer);
    removeParentAbort();
  }
}

/** Bound provider cleanup after a timed-out or abandoned turn; a late close still runs in the background. */
export async function closeProviderSession(session: { close(): Promise<void> }, provider: string,
  timeoutMs = 15_000): Promise<void> {
  const started = Date.now(), startedAt = new Date(started).toISOString();
  report({ phase: 'provider_close', status: 'started', provider, started_at: startedAt,
    deadline_at: new Date(started + timeoutMs).toISOString(), timeout_ms: timeoutMs });
  let timer: ReturnType<typeof setTimeout> | undefined;
  let timedOut = false;
  const close = Promise.resolve().then(() => session.close());
  try {
    await Promise.race([close, new Promise<never>((_resolve, reject) => {
      timer = setTimeout(() => { timedOut = true; reject(new Error('provider cleanup deadline exceeded')); }, timeoutMs);
    })]);
    report({ phase: 'provider_close', status: 'completed', provider, started_at: startedAt,
      ended_at: new Date().toISOString(), elapsed_ms: Date.now() - started });
  } catch (error) {
    report({ phase: 'provider_close', status: timedOut ? 'deadline_exceeded' : 'failed', provider, started_at: startedAt,
      ended_at: new Date().toISOString(), elapsed_ms: Date.now() - started,
      error_name: error instanceof Error ? error.name : 'Error' });
  } finally {
    if (timer) clearTimeout(timer);
  }
  // Promise.race installed a rejection handler on close; prevent a late rejection becoming unhandled.
  void close.catch(() => undefined);
}
