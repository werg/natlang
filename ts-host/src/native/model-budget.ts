/** Shared by all invocations in a task. Reservations prevent concurrent children overspending reply tokens. */
export class ModelBudget {
  private requests = 0;
  private tokens = 0;
  private reserved = 0;
  private waiters = new Set<() => void>();

  constructor(readonly maxRequests?: number, readonly maxTokens?: number) {
    for (const [name, value] of [['maxModelRequests', maxRequests], ['maxModelTokens', maxTokens]] as const)
      if (value !== undefined && (!Number.isSafeInteger(value) || value < 0))
        throw new RangeError(`${name} must be a non-negative safe integer`);
  }

  /** A forced source-text writer performs a request but samples no completion tokens. */
  reserveForcedText(): boolean {
    if (this.maxRequests !== undefined && this.requests >= this.maxRequests) return false;
    this.requests++;
    return true;
  }

  hasRemaining(): boolean {
    return (this.maxRequests === undefined || this.requests < this.maxRequests) &&
      (this.maxTokens === undefined || this.tokens < this.maxTokens);
  }

  /** A reservation is synchronous, so two sibling requests cannot reserve the same tokens. */
  async reserve(requested: number | null, signal?: AbortSignal): Promise<{ limit: number | null; finish: (used?: number) => void } | undefined> {
    if (this.maxRequests !== undefined && this.requests >= this.maxRequests) return;
    let remaining = this.maxTokens === undefined ? null : this.maxTokens - this.tokens - this.reserved;
    while (remaining !== null && remaining <= 0 && this.reserved > 0) {
      await new Promise<void>((resolve, reject) => {
        const wake = () => { this.waiters.delete(wake); signal?.removeEventListener('abort', abort); resolve(); };
        const abort = () => { this.waiters.delete(wake); reject(signal?.reason ?? new Error('model budget wait aborted')); };
        if (signal?.aborted) return abort();
        this.waiters.add(wake); signal?.addEventListener('abort', abort, { once: true });
      });
      if (this.maxRequests !== undefined && this.requests >= this.maxRequests) return;
      remaining = this.maxTokens === undefined ? null : this.maxTokens - this.tokens - this.reserved;
    }
    if (remaining !== null && remaining <= 0) return;
    const limit = remaining === null ? requested : requested === null ? remaining : Math.min(requested, remaining);
    if (limit !== null && limit <= 0) return;
    this.requests++;
    const held = this.maxTokens === undefined ? 0 : limit!;
    this.reserved += held;
    let finished = false;
    return { limit, finish: used => {
      if (finished) throw new Error('model budget reservation finished twice');
      finished = true;
      this.reserved -= held;
      // Missing usage or a failed request consumes its full reservation: do not silently reuse unknown work.
      const charged = used === undefined || !Number.isSafeInteger(used) || used < 0 ? held : used;
      this.tokens += charged;
      for (const wake of this.waiters) wake();
    } };
  }
}
