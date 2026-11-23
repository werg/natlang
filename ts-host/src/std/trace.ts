/**
 * The trace of one combinator run: a record `std:<name>` under the call that ran it, with one `std_stage` event per
 * stage naming the calls the stage made. The combinator's own model calls stay ordinary calls with their own records.
 */
import { currentFrame, runInFrame, type Frame } from '../runtime/context.js';
import { resolveFrame } from '../runtime/runtime.js';

let sequence = 0;

export class StdSpan {
  readonly frame: Frame;
  private readonly events: Record<string, unknown>[] = [];
  private mark: number;
  private readonly started = Date.now();

  constructor(readonly name: string) {
    this.frame = currentFrame() ?? resolveFrame();
    this.mark = this.frame.task.traces.length;
  }

  /** Run a caller's or a shipped function in the frame the combinator was called in. */
  run<T>(fn: () => T): T { return runInFrame(this.frame, fn); }

  /** Close a stage: the calls recorded since the previous stage at this level, and the stage's facts. */
  stage(stage: string, data: Record<string, unknown> = {}): void {
    const traces = this.frame.task.traces, parent = this.frame.parentCallId ?? null;
    const calls = traces.slice(this.mark).filter(trace => (trace.parentCallId ?? null) === parent).map(trace => trace.callId);
    this.mark = traces.length;
    this.events.push({ kind: 'std_stage', call_id: null, combinator: this.name, stage, call_ids: calls, ...data });
  }

  end(outcome: 'done' | 'error', detail: string): void {
    const task = this.frame.task;
    task.record({ callId: `${task.id}/std-${this.name}-${++sequence}`, parentCallId: this.frame.parentCallId ?? null, taskId: task.id,
      definitionId: `std:${this.name}`, name: `std ${this.name}`, outcome, detail,
      events: [{ kind: 'std', call_id: null, combinator: this.name, wall_ms: Date.now() - this.started }, ...this.events] });
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
