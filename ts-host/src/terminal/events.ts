/** Push-driven async event stream for readline, jobs, file watchers, or sockets. */
export class TerminalEventQueue<E> implements AsyncIterable<E> {
  private values: E[] = [];
  private waiters: Array<{ resolve: (result: IteratorResult<E>) => void;
    reject: (error: unknown) => void }> = [];
  private ended = false;
  private failure: unknown = null;

  push(value: E): void {
    if (this.ended) throw new Error('terminal event queue is closed');
    const waiter = this.waiters.shift();
    if (waiter) waiter.resolve({ done: false, value }); else this.values.push(value);
  }
  close(): void {
    if (this.ended) return;
    this.ended = true;
    for (const waiter of this.waiters.splice(0)) waiter.resolve({ done: true, value: undefined });
  }
  fail(error: unknown): void {
    if (this.ended) return;
    this.failure = error; this.ended = true;
    for (const waiter of this.waiters.splice(0)) waiter.reject(error);
  }
  [Symbol.asyncIterator](): AsyncIterator<E> {
    return { next: () => {
      if (this.values.length) return Promise.resolve({ done: false, value: this.values.shift()! });
      if (this.failure) return Promise.reject(this.failure);
      if (this.ended) return Promise.resolve({ done: true, value: undefined });
      return new Promise<IteratorResult<E>>((resolve, reject) => this.waiters.push({ resolve, reject }));
    } };
  }
}
