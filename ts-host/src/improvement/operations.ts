/** Durable, idempotent operation identities and compare-and-swap state publication. */
import { mkdirSync, readFileSync, writeFileSync, renameSync, existsSync, openSync, closeSync, unlinkSync } from 'node:fs';
import { join } from 'node:path';
import { fingerprint } from '../adaptation/identity.js';
export class OperationJournal {
  constructor(readonly directory: string) { mkdirSync(directory, { recursive: true }); }
  read<T>(key: string): { status: 'pending' | 'done'; value?: T } | undefined {
    const path = join(this.directory, fingerprint(key) + '.json');
    return existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : undefined;
  }
  record<T>(key: string, value: T): void { this.write(key, { status: 'done', value }); }
  checkpoint<T>(): { revision: number; value: T } | undefined {
    const path = join(this.directory, 'checkpoint.json');
    return existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : undefined;
  }
  private write(key: string, value: unknown): void {
    const path = join(this.directory, fingerprint(key) + '.json'), temp = path + '.' + process.pid + '.tmp';
    writeFileSync(temp, JSON.stringify(value)); renameSync(temp, path);
  }
  private acquire(lock: string): number {
    for (let attempt = 0; attempt < 2; attempt++) {
      try { const descriptor = openSync(lock, 'wx'); writeFileSync(descriptor, String(process.pid)); return descriptor; }
      catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
        const owner = Number(readFileSync(lock, 'utf8'));
        if (!Number.isSafeInteger(owner) || owner < 1) throw new Error('operation lock has no recoverable owner: ' + lock);
        try { process.kill(owner, 0); }
        catch (probe) { if ((probe as NodeJS.ErrnoException).code === 'ESRCH') { unlinkSync(lock); continue; } throw probe; }
        throw new Error('operation already in flight: ' + lock);
      }
    }
    throw new Error('could not acquire operation lock: ' + lock);
  }
  async run<T>(key: string, operation: () => Promise<T>, reconcile?: () => Promise<T>): Promise<T> {
    const cached = this.read<T>(key);
    if (cached?.status === 'done') return cached.value!;
    const lock = join(this.directory, fingerprint(key) + '.lock');
    const descriptor = this.acquire(lock);
    try {
      const previous = this.read<T>(key);
      if (previous?.status === 'done') return previous.value!;
      if (previous?.status === 'pending' && !reconcile) throw new Error('pending external operation needs reconciliation: ' + key);
      this.write(key, { status: 'pending' });
      const value = await (previous?.status === 'pending' ? reconcile!() : operation());
      this.write(key, { status: 'done', value }); return value;
    } finally { closeSync(descriptor); unlinkSync(lock); }
  }
  commit<T>(expectedRevision: number, value: T): number {
    const path = join(this.directory, 'checkpoint.json'), lock = path + '.lock';
    const descriptor = this.acquire(lock);
    try {
      const previous = existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : { revision: 0 };
      if (previous.revision !== expectedRevision) throw new Error('stale checkpoint revision');
      const revision = expectedRevision + 1, temp = path + '.tmp';
      writeFileSync(temp, JSON.stringify({ revision, value })); renameSync(temp, path); return revision;
    } finally { closeSync(descriptor); unlinkSync(lock); }
  }
}
export type Allocation = { collection: number; modelCalls: number; caseExecutions: number; trainingJobs: number; trainingUpdates: number; confirmation: number; cost: number };
/** One assignment owns capacity. Child batches reserve from this ledger; they cannot mint capacity. */
export class AssignmentBudget {
  readonly used: Allocation = { collection: 0, modelCalls: 0, caseExecutions: 0, trainingJobs: 0, trainingUpdates: 0, confirmation: 0, cost: 0 };
  private attempts = new Map<string, number>();
  private allocations = new Map<string,string>();
  private revision = 0;
  constructor(readonly ceiling: Allocation, readonly reserve: Partial<Allocation> = {}, readonly maxClusterAttempts = 3, private readonly journal?: OperationJournal) {
    for (const value of Object.values(ceiling)) if (!Number.isFinite(value) || value < 0) throw new RangeError('finite assignment ceilings are required');
    this.refresh();
  }
  private refresh(): void {
    const checkpoint = this.journal?.checkpoint<ReturnType<AssignmentBudget['snapshot']>>();
    if (!checkpoint) return;
    if (fingerprint(checkpoint.value.ceiling) !== fingerprint(this.ceiling) || fingerprint(checkpoint.value.reserved) !== fingerprint(this.reserve)) throw new Error('assignment allocation cannot change between batches');
    this.revision = checkpoint.revision; Object.assign(this.used, checkpoint.value.used); this.attempts = new Map();
    this.allocations=new Map(Object.entries(checkpoint.value.allocations??{}));
    for(const [stored,count]of Object.entries(checkpoint.value.attempts)){const cluster=stored.replace(/:[a-f0-9]{12}$/, '');this.attempts.set(cluster,(this.attempts.get(cluster)??0)+count);}
  }
  private persist(): void { if (this.journal) this.revision = this.journal.commit(this.revision, this.snapshot()); }
  allocate(request: Partial<Allocation>, final = false): void {
    this.refresh();
    for (const [key, value] of Object.entries(request) as [keyof Allocation, number][]) {
      if (!Number.isFinite(value) || value < 0 || this.used[key] + value > this.ceiling[key] - (final ? 0 : this.reserve[key] ?? 0)) throw new Error('assignment-budget-exhausted: ' + key);
    }
    for (const [key, value] of Object.entries(request) as [keyof Allocation, number][]) this.used[key] += value;
    this.persist();
  }
  /** Charge the operation identity and its capacity in the same durable checkpoint. */
  allocateOnce(id:string,request:Partial<Allocation>,final=false):void{
    if(!id)throw Error('allocation operation identity required');
    this.refresh();const identity=fingerprint({request,final}),prior=this.allocations.get(id);
    if(prior){if(prior!==identity)throw Error('allocation identity changed: '+id);return;}
    for(const [key,value]of Object.entries(request) as [keyof Allocation,number][]){
      if(!Number.isFinite(value)||value<0||this.used[key]+value>this.ceiling[key]-(final?0:this.reserve[key]??0))throw Error('assignment-budget-exhausted: '+key);
    }
    for(const [key,value]of Object.entries(request) as [keyof Allocation,number][])this.used[key]+=value;
    this.allocations.set(id,identity);this.persist();
  }
  attempt(cluster: string): void {
    this.refresh();
    const count = this.attempts.get(cluster) ?? 0;
    if (count >= this.maxClusterAttempts) throw new Error('coverage-exhausted: ' + cluster);
    this.attempts.set(cluster, count + 1); this.persist();
  }
  snapshot() { return { ceiling: this.ceiling, reserved: this.reserve, used: { ...this.used }, attempts: Object.fromEntries(this.attempts),allocations:Object.fromEntries(this.allocations) }; }
}
