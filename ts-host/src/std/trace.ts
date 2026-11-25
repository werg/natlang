/**
 * The trace of one combinator run: a record `std:<name>` under the call that ran it, with one `std_stage` event per
 * stage naming the calls the stage made. Child calls run inside this span's own frame (`parentCallId === this.callId`),
 * so `natlang traces` nests them under the combinator and concurrent combinators never bleed call IDs across stages.
 */
import { currentFrame, runInFrame, type Frame } from '../runtime/context.js';
import { resolveFrame } from '../runtime/runtime.js';

let sequence = 0;

export class StdSpan {
  readonly parentFrame: Frame;
  readonly callId: string;
  readonly frame: Frame;
  private readonly events: Record<string, unknown>[] = [];
  private mark: number;
  private readonly started = Date.now();

  constructor(readonly name: string) {
    this.parentFrame = currentFrame() ?? resolveFrame();
    this.callId = `${this.parentFrame.task.id}/std-${this.name}-${++sequence}`;
    this.frame = { ...this.parentFrame, parentCallId: this.callId,
      ...(this.parentFrame.spanOf ?? this.parentFrame.parentCallId ? { spanOf: this.parentFrame.spanOf ?? this.parentFrame.parentCallId } : {}) };
    this.mark = this.parentFrame.task.traces.length;
  }

  /** Run a caller's or a shipped function inside this combinator's span frame. */
  run<T>(fn: () => T): T { return runInFrame(this.frame, fn); }

  /** Emit a custom event directly onto this span's trace (for example `decision_escalate`). */
  emit(kind: string, data: Record<string, unknown> = {}): void {
    this.events.push({ kind, call_id: this.callId, combinator: this.name, ...data });
  }

  /** Close a stage: the child calls recorded since the previous stage under this span, and the stage's facts. Returns those calls' IDs. */
  stage(stage: string, data: Record<string, unknown> = {}): string[] {
    const traces = this.parentFrame.task.traces;
    const calls = traces.slice(this.mark).filter(trace => trace.parentCallId === this.callId).map(trace => trace.callId);
    this.mark = traces.length;
    this.events.push({ kind: 'std_stage', call_id: this.callId, combinator: this.name, stage, call_ids: calls, ...data });
    return calls;
  }

  end(outcome: 'done' | 'error', detail: string): void {
    const task = this.parentFrame.task;
    task.record({ callId: this.callId, parentCallId: this.parentFrame.parentCallId ?? null, taskId: task.id,
      definitionId: `std:${this.name}`, name: `std ${this.name}`, outcome, detail,
      events: [{ kind: 'std', call_id: this.callId, combinator: this.name, wall_ms: Date.now() - this.started }, ...this.events] });
  }

  /** Run `body` as the combinator, recording the trace whether it returns or throws. */
  async around<T>(body: () => Promise<T>, detail: (value: T) => string): Promise<T> {
    try {
      const value = await body();
      this.end('done', detail(value));
      return value;
    } catch (error) {
      this.end('error', (error as Error)?.message ?? String(error));
      throw error;
    }
  }
}

/** A `TypeError` with a stable code, the form every std argument error takes. */
export function stdError(code: string, message: string): TypeError {
  return Object.assign(new TypeError(`${code}: ${message}`), { code });
}
