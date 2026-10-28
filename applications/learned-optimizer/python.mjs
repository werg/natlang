// The learned optimizer's outside world in plain JavaScript, so that both the program (index.ts) and its tune services
// (tune.services.mjs, loaded by `natlang tune` without the program's build) use one implementation: the persistent
// Python service process and the adaptation services over it. Types: python.d.mts.
import { spawn } from 'node:child_process';

/** The one persistent Python process behind every service (`python -m natlang_neuralese.tune.learned serve`): JSON
 * requests one per line, answered in order. Adaptation contexts stay loaded across calls; commands run in-process.
 * Started on first use; it holds the event loop only while a request is pending, and ends by `close()` or with its
 * parent (end of input). */
export class PythonService {
  #entry; #child = null; #pending = new Map(); #next = 0; #buffer = ''; #tail = '';
  constructor(entry) { this.#entry = entry; }

  #start() {
    const entry = this.#entry;
    const child = spawn(entry.argv[0], [...entry.argv.slice(1), 'natlang_neuralese.tune.learned', 'serve'],
      { cwd: entry.cwd ?? process.cwd(), env: { ...process.env, ...entry.env }, stdio: ['pipe', 'pipe', 'pipe'] });
    child.stdout.on('data', chunk => {
      this.#buffer += chunk.toString();
      let newline;
      while ((newline = this.#buffer.indexOf('\n')) >= 0) {
        const line = this.#buffer.slice(0, newline);
        this.#buffer = this.#buffer.slice(newline + 1);
        if (!line.trim()) continue;
        const reply = JSON.parse(line), waiting = this.#pending.get(reply.id);
        this.#pending.delete(reply.id);
        this.#hold();
        if (reply.error !== undefined) waiting?.reject(new Error(reply.error)); else waiting?.resolve(reply.result);
      }
    });
    child.stderr.on('data', chunk => { this.#tail = (this.#tail + chunk.toString()).slice(-4000); });
    const fail = error => {
      this.#child = null;
      for (const waiting of this.#pending.values()) waiting.reject(error);
      this.#pending.clear();
    };
    child.once('error', fail);
    child.once('exit', code => fail(new Error(`the learned optimizer service exited with ${code}\n${this.#tail.trim()}`)));
    return this.#child = child;
  }

  /** Keep the process (and its pipes) referenced exactly while replies are awaited. */
  #hold() {
    const child = this.#child;
    if (!child) return;
    const method = this.#pending.size ? 'ref' : 'unref';
    child[method]();
    for (const stream of [child.stdin, child.stdout, child.stderr]) stream[method]?.();
  }

  /** One request: `{ command, spec }` for a package command, or an adaptation service call `{ plan, out, op, ... }`. */
  request(body) {
    const child = this.#child ?? this.#start(), id = this.#next++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, { resolve, reject });
      this.#hold();
      child.stdin.write(`${JSON.stringify({ ...body, id })}\n`);
    });
  }

  command(command, spec) { return this.request({ command, spec }); }

  /** End the process after its pending requests. */
  async close() {
    const child = this.#child;
    if (!child) return;
    child.ref();
    await new Promise(resolve => { child.once('exit', () => resolve()); child.stdin.end(); });
  }
}

const neuraleseId = value => {
  const ref = value?.$neuralese;
  if (!ref || typeof ref.id !== 'string') throw new TypeError('a digest must be a native Neuralese value');
  return ref.id;
};

/** The adaptation services over one prepared program context (`apply.Context`: a tune plan and its output dir). The
 * revision plan's `operations` select what each operator may change; the services carry everything else over.
 * Bundles stay host-side; the program sees their identities and exact support losses. */
export class AdaptationServices {
  #python; #plan; #out; #bundles = new Map();
  constructor(python, plan, out) { this.#python = python; this.#plan = plan; this.#out = out; }
  bundle(id) { const found = this.#bundles.get(id); if (!found) throw new Error(`unknown bundle ${id}`); return found; }
  #call(op, extra) { return this.#python.request({ plan: this.#plan, out: this.#out, op, ...extra }); }
  #summary(result, actor) {
    this.#bundles.set(result.bundle.id, result.bundle);
    return { id: result.bundle.id, supportLoss: result.supportLoss, parent: result.bundle.parent, actor };
  }
  #parent(id) { return id === null || id === undefined ? null : this.bundle(id); }
  optimizer = {
    evidence: async () => {
      const result = await this.#call('evidence', {});
      return { view: result.view, scopes: result.scopes, incomingLoss: result.incomingLoss };
    },
    propose: async request => this.#summary(await this.#call('propose', { parent: this.#parent(request.parent), sample: request.sample,
      seed: request.seed, operations: request.operations, focus: request.focus, keyEvidence: request.keyEvidence,
      digests: (request.digests ?? []).map(item => [item.binding, neuraleseId(item.digest)]) }), 'learned-optimizer'),
    refine: async request => this.#summary(await this.#call('refine', { parent: this.#parent(request.parent), steps: request.steps,
      operations: request.operations }), 'in-bank-refinement'),
    retrieve: async request => this.#summary(await this.#call('retrieve', { parent: this.#parent(request.parent),
      operations: request.operations }), 'gradient-free-mixing'),
  };
  /** The exact support loss of a bundle (null: the incoming one). */
  async supportLoss(id) { return (await this.#call('evaluate', { parent: this.#parent(id) })).supportLoss; }
  /** The release fragment (composed scope adapters and writer states) of the selected bundle. */
  release(id) { return this.#call('release', { parent: this.#parent(id) }); }
  finish(id, history, controller) { return this.#call('finish', { parent: this.#parent(id), history, controller }); }
}

export const optimizerDeclaration = `/** Exact adaptation services: they score, write and compose; they never decide what to revise. */
export function evidence(): Promise<{ view: EvidenceView; scopes: string[]; incomingLoss: number }>;
/** The learned writer's complete bundle from parent (null: the incoming bundle); sample false decodes deterministically.
 * It revises exactly what operations selects (per binding: "reencode", "replace-skill", "weights" or "keep"). */
export function propose(request: { parent: string | null; sample: boolean; seed: number; operations: Record<string, RevisionOperation>; focus: string[]; keyEvidence: Record<string, string[]>; digests: { binding: string; digest: Neuralese<EvidenceDigest> }[] }): Promise<BundleSummary>;
/** Local in-bank refinement of the selected entries from parent by gradient on the support evidence. */
export function refine(request: { parent: string | null; steps: number; operations: Record<string, RevisionOperation> }): Promise<BundleSummary>;
/** Gradient-free mixing over the selected scopes' bank views from parent (the retrieval baseline). */
export function retrieve(request: { parent: string | null; operations: Record<string, RevisionOperation> }): Promise<BundleSummary>;`;
