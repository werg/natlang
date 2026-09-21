/** Mount a versioned generated UI module in an origin-isolated iframe.
 * The module owns presentation mechanics and can only emit declared control
 * events through postMessage. Meaning remains in its bound natlang handlers.
 */
export class GeneratedModuleRenderer {
    constructor(root, dispatch, onDraft = () => {}, onError = () => {}) {
        this.root = root; this.dispatch = dispatch; this.onDraft = onDraft; this.onError = onError;
        this.frame = null; this.listener = null; this.token = '';
    }
    render(module, bindings, drafts = {}) {
        this.close();
        if (!module || typeof module.html !== 'string' || typeof module.script !== 'string' ||
            (module.style !== undefined && typeof module.style !== 'string')) throw new Error('Generated module needs html, script and optional style text');
        if (!bindings || typeof bindings !== 'object' || Array.isArray(bindings)) throw new Error('Generated module needs bindings');
        this.token = crypto.randomUUID();
        const frame = document.createElement('iframe');
        frame.className = 'generated-module'; frame.title = module.title || 'Generated interaction';
        frame.setAttribute('sandbox', 'allow-scripts');
        const initial = JSON.stringify(structuredClone(drafts)).replaceAll('<', '\\u003c');
        const token = JSON.stringify(this.token);
        frame.srcdoc = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data: blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline'"><style>html{color-scheme:light}body{margin:0;font:14px/1.45 system-ui,sans-serif;color:#24322c;background:#fbfcfa}button,input,select,textarea{font:inherit}${module.style ?? ''}</style></head><body>${module.html}<script>const __token=${token};const __drafts=${initial};window.natlang=Object.freeze({drafts:__drafts,emit:(control,value)=>parent.postMessage({channel:'natlang-generated-ui',token:__token,kind:'event',control,value},'*'),draft:(id,value)=>parent.postMessage({channel:'natlang-generated-ui',token:__token,kind:'draft',id,value},'*')});try{${module.script}\n}catch(error){parent.postMessage({channel:'natlang-generated-ui',token:__token,kind:'error',message:String(error)},'*')}</script></body></html>`;
        this.listener = event => {
            if (event.source !== frame.contentWindow || event.data?.channel !== 'natlang-generated-ui' || event.data.token !== this.token) return;
            const message = event.data;
            if (message.kind === 'error') return this.onError(new Error(message.message));
            if (message.kind === 'draft') {
                if (typeof message.id === 'string') this.onDraft(message.id, message.value);
                return;
            }
            if (message.kind !== 'event' || typeof message.control !== 'string') return;
            if (!Object.hasOwn(bindings, message.control)) return this.onError(new Error(`Generated module emitted undeclared control ${message.control}`));
            try { void Promise.resolve(this.dispatch({ id: `module-${crypto.randomUUID()}`, kind: message.control, value: message.value }))
                .catch(error => this.onError(error)); }
            catch (error) { this.onError(error); }
        };
        window.addEventListener('message', this.listener);
        this.frame = frame; this.root.replaceChildren(frame);
    }
    close() {
        if (this.listener) window.removeEventListener('message', this.listener);
        this.listener = null; this.frame?.remove(); this.frame = null;
    }
}
