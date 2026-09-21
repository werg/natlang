/** Session-local recipe registry and retained job outcomes for natlang Fold. */
export class RecipeTerminal {
  constructor(recipes) {
    this.recipes = new Map(recipes.map(recipe => [recipe.id, recipe]));
    if (this.recipes.size !== recipes.length) throw new Error('duplicate recipe IDs');
    this.jobs = new Map();
    this.requests = new Set();
    this.nextJob = 1;
    this.events = [];
    this.listeners = new Set();
  }

  catalog() {
    return [...this.recipes.values()].map(({ id, description }) => ({ id, description }));
  }

  start(requestId, recipeId) {
    if (!requestId || this.requests.has(requestId)) throw new Error(`duplicate request: ${requestId}`);
    const recipe = this.recipes.get(recipeId);
    if (!recipe) throw new Error(`unknown recipe: ${recipeId}`);
    this.requests.add(requestId);
    const id = `job-${this.nextJob++}`;
    const job = { id, requestId, recipeId, cancelRequested: false, result: null, promise: null,
      controller: new AbortController() };
    this.jobs.set(id, job);
    this.events.push({ operation: 'terminal.start', request_id: requestId, job_id: id, recipe_id: recipeId });
    job.promise = Promise.resolve().then(() => recipe.run({ signal: job.controller.signal,
      requestId, jobId: id })).then(result => {
      const status = ['ok', 'failed', 'unknown'].includes(result?.status) ? result.status : 'unknown';
      job.result = { status, detail: String(result?.detail ?? '') };
      return this.complete(job);
    }, error => {
      job.result = { status: 'unknown', detail: error instanceof Error ? error.message : String(error) };
      return this.complete(job);
    });
    return { id, request_id: requestId, status: 'running', detail: '' };
  }

  cancel(jobId) {
    const job = this.jobs.get(jobId);
    if (!job) throw new Error(`unknown job: ${jobId}`);
    job.cancelRequested = true;
    job.controller.abort();
    this.events.push({ operation: 'terminal.cancel-requested', job_id: jobId });
    return { id: jobId, request_id: job.requestId, status: 'cancel-requested', detail: '' };
  }

  async wait(requestId) {
    const job = [...this.jobs.values()].find(row => row.requestId === requestId);
    if (!job) throw new Error(`no job for request: ${requestId}`);
    await job.promise;
    return { ...job.completion };
  }

  complete(job) {
    const result = job.result;
    const detail = job.cancelRequested ? `Cancellation was requested. Actual result: ${result.detail}` : result.detail;
    this.events.push({ operation: 'terminal.complete', request_id: job.requestId, job_id: job.id,
      status: result.status, cancel_requested: job.cancelRequested });
    job.completion = { kind: 'complete', id: `completion-${job.id}`, request_id: job.requestId,
      job_id: job.id, text: '', status: result.status, detail };
    for (const listener of this.listeners) {
      try { listener({ ...job.completion }); }
      catch (error) { this.events.push({ operation: 'terminal.listener-failed', job_id: job.id,
        detail: error instanceof Error ? error.message : String(error) }); }
    }
    return result;
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  close() {
    for (const job of this.jobs.values()) if (!job.result) job.controller.abort();
    this.listeners.clear();
  }

  confirm(event) {
    const job = this.jobs.get(event.job_id);
    return !!job?.completion && job.completion.request_id === event.request_id &&
      job.completion.status === event.status && job.completion.detail === event.detail;
  }

  drainEvents() { return this.events.splice(0); }
}
