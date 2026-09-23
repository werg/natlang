import type { AppEvent } from '../app/event-loop.js';

export type UiAction = { kind: string; value?: string; from?: string };
export type UiNode = { tag: string; text?: string; value?: string; id?: string;
  label?: string; placeholder?: string; disabled?: boolean;
  max?: number; min?: number; action?: UiAction; children?: UiNode[] };

const TAGS = new Set(['main', 'section', 'div', 'h1', 'h2', 'h3', 'p', 'span',
  'strong', 'em', 'ul', 'ol', 'li', 'button', 'input', 'label', 'output',
  'table', 'thead', 'tbody', 'tr', 'th', 'td', 'textarea', 'select', 'option',
  'details', 'summary', 'pre', 'code', 'meter', 'progress', 'br']);

/** Optional DOM projection; natlang supplies data, never raw HTML or JavaScript. */
export class BrowserDomRenderer {
  private nextEvent = 0;
  private disposed = false;
  constructor(private readonly root: HTMLElement,
    private readonly dispatch: (event: AppEvent & { value?: string }) => Promise<unknown> | unknown,
    private readonly onError?: (error: unknown) => void) {}

  render(node: UiNode): void {
    if (this.disposed) throw new Error('renderer is closed');
    const inputs = new Map<string, HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>();
    const sources: string[] = [];
    const seen = new WeakSet<UiNode>();
    let next: HTMLElement | undefined;
    const pending: Array<{ item: UiNode; parent?: HTMLElement; finish?: HTMLElement }> = [{ item: node }];
    while (pending.length) {
      const { item, parent, finish } = pending.pop()!;
      if (finish) {
        if (item.value !== undefined) (finish as HTMLSelectElement).value = item.value;
        continue;
      }
      if (!item || !TAGS.has(item.tag)) throw new Error(`unsupported view tag: ${item?.tag}`);
      if (seen.has(item)) throw new Error('view tree reuses a node');
      seen.add(item);
      const element = document.createElement(item.tag);
      if (parent) parent.appendChild(element);
      else next = element;
      if (item.id) element.id = item.id;
      if (item.text !== undefined) element.textContent = item.text;
      if (item.tag === 'input' || item.tag === 'textarea' || item.tag === 'select') {
        const input = element as HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement;
        if (item.tag === 'input') (input as HTMLInputElement).type = 'text';
        input.value = item.value ?? '';
        if (item.tag !== 'select') input.setAttribute('placeholder', item.placeholder ?? '');
        input.setAttribute('aria-label', item.label ?? item.placeholder ?? 'Input');
        if (item.id) {
          if (inputs.has(item.id)) throw new Error(`duplicate input ID: ${item.id}`);
          inputs.set(item.id, input);
        }
      }
      if (item.tag === 'button' || item.tag === 'input' || item.tag === 'select' || item.tag === 'textarea')
        (element as HTMLButtonElement).disabled = item.disabled ?? false;
      if (item.tag === 'button') (element as HTMLButtonElement).type = 'button';
      if (item.tag === 'option' && item.value !== undefined) (element as HTMLOptionElement).value = item.value;
      if (item.tag === 'meter' || item.tag === 'progress') {
        const meter = element as HTMLMeterElement;
        const numeric = Number(item.value ?? 0);
        if (!Number.isFinite(numeric)) throw new Error('meter/progress value must be finite');
        meter.value = numeric;
        if (item.max !== undefined) meter.max = item.max;
        if (item.min !== undefined && item.tag === 'meter') meter.min = item.min;
      }
      if (item.action) {
        if (!['button', 'input', 'select', 'textarea'].includes(item.tag))
          throw new Error('only controls can emit events');
        if (!item.action.kind) throw new Error('event action needs a kind');
        if (item.action.from) sources.push(item.action.from);
        const name = item.tag === 'button' ? 'click' : 'change';
        element.addEventListener(name, () => {
          if (this.disposed) return;
          const value = item.action!.from ? inputs.get(item.action!.from)?.value :
            item.tag !== 'button' ? (element as HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement).value : item.action!.value;
          try {
            void Promise.resolve(this.dispatch({ id: `ui-${++this.nextEvent}`,
              kind: item.action!.kind, ...(value === undefined ? {} : { value }) }))
              .catch(error => this.onError?.(error));
          } catch (error) { this.onError?.(error); }
        });
      }
      if (item.children !== undefined && !Array.isArray(item.children)) throw new Error('view children must be a list');
      if (item.tag === 'select') pending.push({ item, finish: element });
      for (const child of [...(item.children ?? [])].reverse()) pending.push({ item: child, parent: element });
    }
    if (sources.some(id => !inputs.has(id))) throw new Error('action references unknown input');
    this.root.replaceChildren(next!);
  }

  close(): void { this.disposed = true; this.root.replaceChildren(); }
}
