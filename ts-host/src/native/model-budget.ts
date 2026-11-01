/** Task-wide model budget. Reservations serialize contending graph work without lending the same allowance twice. */
export class ModelBudget {
  private requests = 0;
  private tokens = 0;
  private reservedRequests = 0;
  private reservedTokens = 0;
  private requestsUnknown = false;
  private tokensUnknown = false;
  private waiters = new Set<() => void>();

  constructor(readonly maxRequests?: number, readonly maxTokens?: number) {
    for (const [name, value] of [['maxModelRequests', maxRequests], ['maxModelTokens', maxTokens]] as const)
      if (value !== undefined && (!Number.isSafeInteger(value) || value < 0))
        throw new RangeError(`${name} must be a non-negative safe integer`);
  }

  private async wait(signal?: AbortSignal): Promise<void> {
    await new Promise<void>((resolve, reject) => {
      const wake = () => { this.waiters.delete(wake); signal?.removeEventListener('abort', abort); resolve(); };
      const abort = () => { this.waiters.delete(wake); reject(signal?.reason ?? new Error('model budget wait aborted')); };
      if (signal?.aborted) return abort();
      this.waiters.add(wake); signal?.addEventListener('abort', abort, { once: true });
    });
  }
  private wake(): void { for (const wake of this.waiters) wake(); }
  hasRemaining(): boolean {
    return (this.maxRequests === undefined || this.requests < this.maxRequests) &&
      (this.maxTokens === undefined || this.tokens < this.maxTokens);
  }
  /** Null records unavailable usage; zero always means a measured zero. */
  usage(): { requests: number | null; tokens: number | null } {
    return { requests: this.requestsUnknown ? null : this.requests, tokens: this.tokensUnknown ? null : this.tokens };
  }

  /** A forced source-text writer performs a request but samples no completion tokens. */
  async reserveForcedText(signal?: AbortSignal): Promise<boolean> {
    while (this.maxRequests !== undefined && this.requests + this.reservedRequests >= this.maxRequests) {
      if (this.requests >= this.maxRequests) return false;
      await this.wait(signal);
    }
    this.requests++;
    return true;
  }

  /** Reserve one request and its maximum completion tokens. */
  async reserve(requested: number | null, signal?: AbortSignal): Promise<{ limit: number | null; finish: (used?: number) => void } | undefined> {
    while (true) {
      if (!this.hasRemaining()) return;
      const requests = this.maxRequests === undefined ? null : this.maxRequests - this.requests - this.reservedRequests;
      const tokens = this.maxTokens === undefined ? null : this.maxTokens - this.tokens - this.reservedTokens;
      if ((requests === null || requests > 0) && (tokens === null || tokens > 0)) break;
      await this.wait(signal);
    }
    const remaining = this.maxTokens === undefined ? null : this.maxTokens - this.tokens - this.reservedTokens;
    const limit = remaining === null ? requested : requested === null ? remaining : Math.min(requested, remaining);
    if (limit !== null && limit <= 0) return;
    this.requests++;
    const held = this.maxTokens === undefined ? 0 : limit!;
    this.reservedTokens += held;
    let finished = false;
    return { limit, finish: used => {
      if (finished) throw new Error('model budget reservation finished twice');
      finished = true;
      this.reservedTokens -= held;
      const valid = used !== undefined && Number.isSafeInteger(used) && used >= 0;
      if (!valid) this.tokensUnknown = true;
      this.tokens += valid ? used : held;
      this.wake();
    } };
  }

  /** Reserve the remaining graph allowance for a multi-write operation such as native view. */
  async reserveBatch(signal?: AbortSignal): Promise<{ maxRequests: number | null; maxTokens: number | null;
    finish: (usage?: { requests: number; tokens: number }) => void } | undefined> {
    while (true) {
      if (!this.hasRemaining()) return;
      const requests = this.maxRequests === undefined ? null : this.maxRequests - this.requests - this.reservedRequests;
      const tokens = this.maxTokens === undefined ? null : this.maxTokens - this.tokens - this.reservedTokens;
      if ((requests === null || requests > 0) && (tokens === null || tokens > 0)) {
        const heldRequests = requests ?? 0, heldTokens = tokens ?? 0;
        this.reservedRequests += heldRequests; this.reservedTokens += heldTokens;
        let finished = false;
        return { maxRequests: requests, maxTokens: tokens, finish: usage => {
          if (finished) throw new Error('model budget reservation finished twice');
          finished = true;
          this.reservedRequests -= heldRequests; this.reservedTokens -= heldTokens;
          const valid = usage && Number.isSafeInteger(usage.requests) && usage.requests >= 0 &&
            Number.isSafeInteger(usage.tokens) && usage.tokens >= 0;
          if (!valid) { this.requestsUnknown = true; this.tokensUnknown = true; }
          this.requests += valid ? usage.requests : heldRequests;
          this.tokens += valid ? usage.tokens : heldTokens;
          this.wake();
        } };
      }
      await this.wait(signal);
    }
  }
}
