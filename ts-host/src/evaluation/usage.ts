import type { ModelDriver } from '../runtime/runtime.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import type { BudgetLimits, Usage } from './types.js';
export class BudgetExhausted extends Error { constructor(readonly dimension: string) { super('adaptation budget exhausted: ' + dimension); this.name = 'BudgetExhausted'; } }
export type ModelRole = 'executor' | 'reflection' | 'judge';
export type RequestReservation = Partial<Record<'modelCalls' | 'inputTokens' | 'outputTokens' | 'cost', number>>;
export type BudgetLedger = { rollouts: number; proposals: number; usage: Usage; elapsedMs: number; unknownRequests: number; overrun: readonly string[];
  roles?: Partial<Record<ModelRole, Usage>>;
  reserved?: { modelCalls: number; inputTokens: number; outputTokens: number; cost: number } };
export class UsageGateway {
  readonly ledger: BudgetLedger;
  private readonly started = Date.now();
  private inFlight = 0;
  private reservedInput = 0;
  private reservedOutput = 0;
  private reservedCost = 0;
  private protectedCapacity: RequestReservation = {};
  protect(capacity: RequestReservation): void {
    if (Object.values(capacity).some(value => !Number.isFinite(value) || value! < 0)) throw new Error('invalid holdout request reservation');
    this.protectedCapacity = { ...capacity };
  }
  onUpdate?: (ledger: BudgetLedger) => void;
  constructor(readonly limits: BudgetLimits, previous?: BudgetLedger) {
    this.ledger = previous ? structuredClone(previous) : { rollouts: 0, proposals: 0,
      usage: { modelCalls: 0, inputTokens: 0, outputTokens: 0, cost: null }, elapsedMs: 0, unknownRequests: 0, overrun: [] };
    if (limits.maxCost !== undefined && (!limits.pricing || !limits.requestBounds)) throw new Error('hard monetary caps require configured pricing and bounded requests');
    this.ledger.roles ??= {};
    // A crashed coordinator cannot know what the provider billed for outstanding requests.
    if (previous?.reserved?.modelCalls) {
      this.ledger.unknownRequests += previous.reserved.modelCalls;
      this.ledger.usage.inputTokens = this.ledger.usage.outputTokens = this.ledger.usage.cost = null;
      for (const use of Object.values(this.ledger.roles)) if (use) use.inputTokens = use.outputTokens = use.cost = null;
    }
    delete this.ledger.reserved;
    if (!previous && limits.pricing) this.ledger.usage.cost = 0;
  }
  check(signal?: AbortSignal): void {
    signal?.throwIfAborted();
    if (this.limits.maxElapsedMs !== undefined && this.elapsed() >= this.limits.maxElapsedMs) throw new BudgetExhausted('elapsedMs');
    if (this.limits.maxInputTokens !== undefined && (this.ledger.usage.inputTokens === null || this.ledger.usage.inputTokens >= this.limits.maxInputTokens)) throw new BudgetExhausted('inputTokens');
    if (this.limits.maxOutputTokens !== undefined && (this.ledger.usage.outputTokens === null || this.ledger.usage.outputTokens >= this.limits.maxOutputTokens)) throw new BudgetExhausted('outputTokens');
    if (this.limits.maxCost !== undefined && (this.ledger.usage.cost === null || this.ledger.usage.cost >= this.limits.maxCost)) throw new BudgetExhausted('cost');
  }
  elapsed(): number { return this.ledger.elapsedMs + Date.now() - this.started; }
  reserve(kind: 'rollouts' | 'proposals', count = 1, signal?: AbortSignal): void {
    this.check(signal); const max = kind === 'rollouts' ? this.limits.maxRollouts : this.limits.maxProposals;
    if (this.ledger[kind] + count > max) throw new BudgetExhausted(kind); this.ledger[kind] += count; this.onUpdate?.(this.snapshot());
  }
  async request(driver: ModelDriver, request: ModelTurnRequest, signal?: AbortSignal, role: ModelRole = 'executor'): Promise<ModelTurn> {
    this.check(signal); if (this.ledger.usage.modelCalls >= this.limits.maxModelCalls - (this.protectedCapacity.modelCalls ?? 0)) throw new BudgetExhausted('modelCalls');
    const bounds = this.limits.requestBounds;
    const inputBound = bounds?.maxInputTokens ?? 0;
    const outputBound = bounds ? Math.min(request.max_tokens ?? bounds.maxOutputTokens, bounds.maxOutputTokens) : 0;
    const costBound = this.cost(inputBound, outputBound) ?? 0;
    if (bounds) {
      if (this.limits.maxInputTokens !== undefined && this.ledger.usage.inputTokens! + this.reservedInput + inputBound > this.limits.maxInputTokens - (this.protectedCapacity.inputTokens ?? 0)) throw new BudgetExhausted('inputTokens (reserved)');
      if (this.limits.maxOutputTokens !== undefined && this.ledger.usage.outputTokens! + this.reservedOutput + outputBound > this.limits.maxOutputTokens - (this.protectedCapacity.outputTokens ?? 0)) throw new BudgetExhausted('outputTokens (reserved)');
      if (this.limits.maxCost !== undefined && this.ledger.usage.cost! + this.reservedCost + costBound > this.limits.maxCost - (this.protectedCapacity.cost ?? 0)) throw new BudgetExhausted('cost (reserved)');
      request = { ...request, max_tokens: outputBound };
    }
    const roleUsage = this.ledger.roles![role] ??= { modelCalls: 0, inputTokens: 0, outputTokens: 0, cost: this.limits.pricing ? 0 : null };
    roleUsage.modelCalls++;
    this.reservedInput += inputBound; this.reservedOutput += outputBound; this.reservedCost += costBound;
    this.ledger.usage.modelCalls++; this.inFlight++; this.onUpdate?.(this.snapshot());
    try {
      let stop: (() => void) | undefined;
      const abort = new Promise<never>((_resolve, reject) => {
        stop = () => reject(signal?.reason instanceof Error ? signal.reason : new Error('model request cancelled; billed usage unknown'));
        signal?.addEventListener('abort', stop, { once: true });
      });
      let result: ModelTurn;
      try {
        // Defer invoking the driver until its abort listener is installed. A driver may
        // synchronously trigger cancellation while opening a request.
        const response = Promise.resolve().then(() => driver(request, signal));
        if (signal?.aborted) stop?.();
        result = await Promise.race([response, abort]);
      }
      finally { if (stop) signal?.removeEventListener('abort', stop); }
      for (const [field, reported] of [['inputTokens', result.prompt_tokens], ['outputTokens', result.completion_tokens]] as const) {
        if (reported === undefined) this.ledger.usage[field] = null;
        else if (!Number.isSafeInteger(reported) || reported < 0) throw new Error('malformed model usage');
        else if (this.ledger.usage[field] !== null) this.ledger.usage[field]! += reported;
        if (reported === undefined) roleUsage[field] = null;
        else if (roleUsage[field] !== null) roleUsage[field]! += reported;
      }
      const actualCost = result.prompt_tokens === undefined || result.completion_tokens === undefined ? null : this.cost(result.prompt_tokens, result.completion_tokens);
      if (actualCost === null) this.ledger.usage.cost = roleUsage.cost = null;
      else { if (this.ledger.usage.cost !== null) this.ledger.usage.cost += actualCost; if (roleUsage.cost !== null) roleUsage.cost += actualCost; }
      return result;
    } catch (error) {
      this.ledger.unknownRequests++; this.ledger.usage.inputTokens = this.ledger.usage.outputTokens = this.ledger.usage.cost = null;
      roleUsage.inputTokens = roleUsage.outputTokens = roleUsage.cost = null; throw error;
    }
    finally {
      this.inFlight--; this.reservedInput -= inputBound; this.reservedOutput -= outputBound; this.reservedCost -= costBound;
      this.onUpdate?.(this.snapshot());
    }
  }
  private cost(input: number, output: number): number | null {
    return this.limits.pricing ? (input * this.limits.pricing.inputPerMillion + output * this.limits.pricing.outputPerMillion) / 1_000_000 : null;
  }
  snapshot(): BudgetLedger {
    const overrun: string[] = [];
    if (this.limits.maxElapsedMs !== undefined && this.elapsed() > this.limits.maxElapsedMs) overrun.push('elapsedMs');
    if (this.limits.maxInputTokens !== undefined && (this.ledger.usage.inputTokens ?? 0) > this.limits.maxInputTokens) overrun.push('inputTokens');
    if (this.limits.maxOutputTokens !== undefined && (this.ledger.usage.outputTokens ?? 0) > this.limits.maxOutputTokens) overrun.push('outputTokens');
    if (this.limits.maxCost !== undefined && (this.ledger.usage.cost ?? 0) > this.limits.maxCost) overrun.push('cost');
    return structuredClone({ ...this.ledger, elapsedMs: this.elapsed(), overrun,
      reserved: { modelCalls: this.inFlight, inputTokens: this.reservedInput, outputTokens: this.reservedOutput, cost: this.reservedCost } });
  }
}
