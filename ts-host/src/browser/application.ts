import type { BrowserNatlangClient, BrowserClientRun } from './client.js';
import type { BrowserRunRequest } from './host.js';

export type BrowserAppEvent = { id: string; kind: string; value?: string };
export type BrowserAppSource = { files: Record<string, string>; reducer: string; view: string };
export type BrowserAppTransition<S, V> = { event: BrowserAppEvent | null;
  revision: number; state: S; view: V; reducerRun: BrowserClientRun | null;
  viewRun: BrowserClientRun };
export type BrowserAppFailure = { event: BrowserAppEvent | null;
  stage: 'reduce' | 'view'; revision: number; detail: string;
  run: BrowserClientRun | null };
export type BrowserAppCommit<S> = { event: BrowserAppEvent; revision: number; state: S; reducerRun: BrowserClientRun };
export type BrowserAppOptions<S, V> = { client: Pick<BrowserNatlangClient, 'run'>;
  source: BrowserAppSource; initialState: S;
  initialRevision?: number;
  /** Persist a completed reduction before publishing it or computing its view. */
  onCommit?: (commit: BrowserAppCommit<S>) => void | Promise<void>;
  /** Each completed event step is equivalent to one Fold reduction. */
  onTransition?: (transition: BrowserAppTransition<S, V>) => void;
  onFailure?: (failure: BrowserAppFailure) => void;
  seedRoot?: number;
  runOptions?: Omit<NonNullable<BrowserRunRequest['options']>, 'seed'>;
  modelTurn?: BrowserRunRequest['modelTurn'] };

function eventSeed(root: number, revision: number, id: string): number {
  let value = root >>> 0;
  for (const char of `${revision}:${id}`) {
    value ^= char.charCodeAt(0);
    value = Math.imul(value, 16777619) >>> 0;
  }
  return value;
}

/** Serial UI event stream over a natlang reducer and a natlang view function. */
export class BrowserNatlangApplication<S, V> {
  private readonly client: Pick<BrowserNatlangClient, 'run'>;
  private readonly source: BrowserAppSource;
  private readonly options: BrowserAppOptions<S, V>;
  private readonly seen = new Set<string>();
  private queue: Promise<unknown> = Promise.resolve();
  private active: AbortController | null = null;
  private started = false;
  private closed = false;
  private stateValue: S;
  private viewValue: V | null = null;
  private revisionValue = 0;

  constructor(options: BrowserAppOptions<S, V>) {
    if (!options.source.files[options.source.reducer] || !options.source.files[options.source.view])
      throw new Error('application reducer and view source must be present');
    if (options.seedRoot !== undefined && !Number.isSafeInteger(options.seedRoot))
      throw new Error('seedRoot must be a safe integer');
    if (options.initialRevision !== undefined && (!Number.isSafeInteger(options.initialRevision) || options.initialRevision < 0))
      throw new Error('initialRevision must be a nonnegative safe integer');
    this.revisionValue = options.initialRevision ?? 0;
    this.options = options;
    this.client = options.client;
    this.source = { ...options.source, files: { ...options.source.files } };
    this.stateValue = structuredClone(options.initialState);
  }

  get state(): S { return structuredClone(this.stateValue); }
  get view(): V | null { return this.viewValue === null ? null : structuredClone(this.viewValue); }
  get revision(): number { return this.revisionValue; }

  private enqueue<T>(task: () => Promise<T>): Promise<T> {
    if (this.closed) return Promise.reject(new Error('application is closed'));
    const execute = () => {
      if (this.closed) throw new Error('application is closed');
      return task();
    };
    const next = this.queue.then(execute, execute);
    this.queue = next.catch(() => undefined);
    return next;
  }

  private async run(root: string, inputs: Record<string, unknown>,
    revision: number, eventId: string): Promise<BrowserClientRun> {
    const controller = new AbortController();
    this.active = controller;
    try {
      return await this.client.run({
        source: { kind: 'files', root, files: this.source.files }, inputs,
        modelTurn: this.options.modelTurn, signal: controller.signal,
        options: { ...this.options.runOptions,
          ...(this.options.seedRoot === undefined ? {} : {
            seed: { mode: 'derived', root: eventSeed(this.options.seedRoot, revision, eventId) },
          }) },
      });
    } finally { if (this.active === controller) this.active = null; }
  }

  private fail(event: BrowserAppEvent | null, stage: 'reduce' | 'view',
    run: BrowserClientRun | null, detail: string): never {
    this.options.onFailure?.({ event, stage, revision: this.revisionValue, detail, run });
    throw new Error(`${stage} failed: ${detail}`);
  }

  private async render(event: BrowserAppEvent | null,
    reducerRun: BrowserClientRun | null): Promise<BrowserAppTransition<S, V>> {
    let viewRun: BrowserClientRun;
    try { viewRun = await this.run(this.source.view, { state: structuredClone(this.stateValue) },
      this.revisionValue, event?.id ?? 'initial-view'); }
    catch (error) { return this.fail(event, 'view', null, String(error)); }
    if (viewRun.outcome.kind !== 'done')
      return this.fail(event, 'view', viewRun, viewRun.outcome.detail);
    this.viewValue = structuredClone(viewRun.value as V);
    const transition = { event, revision: this.revisionValue,
      state: this.state, view: this.view, reducerRun, viewRun } as BrowserAppTransition<S, V>;
    try { this.options.onTransition?.(transition); }
    catch (error) { return this.fail(event, 'view', viewRun, String(error)); }
    return transition;
  }

  start(): Promise<BrowserAppTransition<S, V>> {
    return this.enqueue(async () => {
      if (this.started) throw new Error('application already started');
      this.started = true;
      // A failed presentation does not invalidate the initial state or lifecycle.
      // The caller can refresh the view or dispatch an event after correcting it.
      return this.render(null, null);
    });
  }

  dispatch(event: BrowserAppEvent): Promise<BrowserAppTransition<S, V> | null> {
    if (!event || !event.id || !event.kind) return Promise.reject(new Error('event needs id and kind'));
    return this.enqueue(async () => {
      if (!this.started) throw new Error('application has not started');
      if (this.seen.has(event.id)) return null;
      let reducerRun: BrowserClientRun;
      try { reducerRun = await this.run(this.source.reducer, {
        state: structuredClone(this.stateValue), event: structuredClone(event),
      }, this.revisionValue, event.id); }
      catch (error) { return this.fail(event, 'reduce', null, String(error)); }
      if (reducerRun.outcome.kind !== 'done')
        return this.fail(event, 'reduce', reducerRun, reducerRun.outcome.detail);
      const state = structuredClone(reducerRun.value as S);
      try { await this.options.onCommit?.({ event, revision: this.revisionValue + 1,
        state: structuredClone(state), reducerRun }); }
      catch (error) { return this.fail(event, 'reduce', reducerRun, `commit failed: ${String(error)}`); }
      this.stateValue = state;
      this.revisionValue++;
      this.seen.add(event.id);
      return this.render(event, reducerRun);
    });
  }

  /** Retry presentation without re-executing a committed event or host effects. */
  refresh(): Promise<BrowserAppTransition<S, V>> {
    return this.enqueue(async () => {
      if (!this.started) throw new Error('application has not started');
      return this.render(null, null);
    });
  }

  cancel(): void { this.active?.abort(); }

  async consume(events: AsyncIterable<BrowserAppEvent>): Promise<void> {
    for await (const event of events) await this.dispatch(event);
  }

  async close(): Promise<void> {
    this.closed = true;
    this.active?.abort();
    await this.queue;
  }
}

export type UiAction = { kind: string; value?: string; from?: string };
export type UiNode = { tag: string; text?: string; value?: string; id?: string;
  label?: string; placeholder?: string; disabled?: boolean;
  action?: UiAction; children?: UiNode[] };

const TAGS = new Set(['main', 'section', 'div', 'h1', 'h2', 'h3', 'p', 'span',
  'strong', 'em', 'ul', 'ol', 'li', 'button', 'input', 'label', 'output']);

/** Optional DOM projection; natlang supplies data, never raw HTML or JavaScript. */
export class BrowserDomRenderer {
  private nextEvent = 0;
  private disposed = false;
  constructor(private readonly root: HTMLElement,
    private readonly dispatch: (event: BrowserAppEvent) => Promise<unknown> | unknown,
    private readonly onError?: (error: unknown) => void) {}

  render(node: UiNode): void {
    if (this.disposed) throw new Error('renderer is closed');
    let count = 0;
    const inputs = new Map<string, HTMLInputElement>();
    const sources: string[] = [];
    const build = (item: UiNode, depth: number): HTMLElement => {
      if (++count > 2000 || depth > 32) throw new Error('view exceeds renderer bounds');
      if (!item || !TAGS.has(item.tag)) throw new Error(`unsupported view tag: ${item?.tag}`);
      const element = document.createElement(item.tag);
      if (item.id) element.id = item.id;
      if (item.text !== undefined) element.textContent = item.text;
      if (item.tag === 'input') {
        const input = element as HTMLInputElement;
        input.type = 'text';
        input.value = item.value ?? '';
        input.placeholder = item.placeholder ?? '';
        input.setAttribute('aria-label', item.label ?? item.placeholder ?? 'Input');
        if (item.id) {
          if (inputs.has(item.id)) throw new Error(`duplicate input ID: ${item.id}`);
          inputs.set(item.id, input);
        }
      }
      if (item.tag === 'button' || item.tag === 'input')
        (element as HTMLButtonElement).disabled = item.disabled ?? false;
      if (item.action) {
        if (item.tag !== 'button' && item.tag !== 'input')
          throw new Error('only controls can emit events');
        if (!item.action.kind) throw new Error('event action needs a kind');
        if (item.action.from) sources.push(item.action.from);
        const name = item.tag === 'input' ? 'change' : 'click';
        element.addEventListener(name, () => {
          if (this.disposed) return;
          const value = item.action!.from ? inputs.get(item.action!.from)?.value :
            item.tag === 'input' ? (element as HTMLInputElement).value : item.action!.value;
          try {
            void Promise.resolve(this.dispatch({ id: `ui-${++this.nextEvent}`,
              kind: item.action!.kind, ...(value === undefined ? {} : { value }) }))
              .catch(error => this.onError?.(error));
          } catch (error) { this.onError?.(error); }
        });
      }
      for (const child of item.children ?? []) element.appendChild(build(child, depth + 1));
      return element;
    };
    const next = build(node, 0);
    if (sources.some(id => !inputs.has(id))) throw new Error('action references unknown input');
    this.root.replaceChildren(next);
  }

  close(): void { this.disposed = true; this.root.replaceChildren(); }
}
