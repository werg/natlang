/** Session-local recipe registry and retained job outcomes for natlang Fold. */
export class RecipeTerminal {
  constructor(recipes) {
    this.recipes = new Map(recipes.map(recipe => [recipe.id, recipe]));
    if (this.recipes.size !== recipes.length) throw new Error('duplicate recipe IDs');
    this.jobs = new Map();
    this.requests = new Set();
    this.nextJob = 1;
    this.events = [];
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
    const job = { id, requestId, recipeId, cancelRequested: false, result: null, promise: null };
    this.jobs.set(id, job);
    this.events.push({ operation: 'terminal.start', request_id: requestId, job_id: id, recipe_id: recipeId });
    job.promise = Promise.resolve().then(() => recipe.run()).then(result => {
      const status = ['ok', 'failed', 'unknown'].includes(result?.status) ? result.status : 'unknown';
      job.result = { status, detail: String(result?.detail ?? '') };
      return job.result;
    }, error => {
      job.result = { status: 'unknown', detail: error instanceof Error ? error.message : String(error) };
      return job.result;
    });
    return { id, request_id: requestId, status: 'running', detail: '' };
  }

  cancel(jobId) {
    const job = this.jobs.get(jobId);
    if (!job) throw new Error(`unknown job: ${jobId}`);
    job.cancelRequested = true;
    this.events.push({ operation: 'terminal.cancel-requested', job_id: jobId });
    return { id: jobId, request_id: job.requestId, status: 'cancel-requested', detail: '' };
  }

  async wait(requestId) {
    const job = [...this.jobs.values()].find(row => row.requestId === requestId);
    if (!job) throw new Error(`no job for request: ${requestId}`);
    const result = await job.promise;
    const detail = job.cancelRequested ? `Cancellation was requested. Actual result: ${result.detail}` : result.detail;
    this.events.push({ operation: 'terminal.complete', request_id: requestId, job_id: job.id,
      status: result.status, cancel_requested: job.cancelRequested });
    job.completion = { kind: 'complete', id: '', request_id: requestId, job_id: job.id, text: '',
      status: result.status, detail };
    return { ...job.completion };
  }

  confirm(event) {
    const job = this.jobs.get(event.job_id);
    return !!job?.completion && job.completion.request_id === event.request_id &&
      job.completion.status === event.status && job.completion.detail === event.detail;
  }

  drainEvents() { return this.events.splice(0); }
}
