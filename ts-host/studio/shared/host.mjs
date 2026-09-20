import { clone } from './domain.mjs';
/** Host mechanics for one operation. Sequencing and semantic choices remain in natlang. */
export async function applyOperation(spec, state, decision, services) {
    try {
        const next = await spec.apply(clone(state), decision, services);
        return { state: next, ok: true, detail: next.notice };
    }
    catch (error) {
        return { state: { ...clone(state), notice: String(error) }, ok: false, detail: String(error) };
    }
}
export class Companion {
    constructor({ progress = () => { }, session = localStorage.getItem('natlang-studio-session') } = {}) { this.progress = progress; this.session = session; this.token = null; this.active = null; }
    async connect() { const response = await fetch('/api/studio/session?session=' + encodeURIComponent(this.session ?? '')); if (!response.ok)
        throw new Error('Start the local companion: node ts-host/scripts/serve-studio.mjs'); const value = await response.json(); this.token = value.token; this.session = value.session; localStorage.setItem('natlang-studio-session', this.session); }
    async request(path, options = {}) { if (!this.token)
        await this.connect(); const response = await fetch('/api/studio/' + path, { ...options, headers: { 'x-studio-token': this.token, 'x-studio-session': this.session, ...options.headers } }); const body = await response.json(); if (!response.ok)
        throw new Error(body.error ?? `HTTP ${response.status}`); return body; }
    async run(operation, payload, id, signal) {
        if (signal?.aborted)
            throw new Error('Operation cancelled');
        const job = await this.request('jobs', { method: 'POST', body: JSON.stringify({ id, operation, payload }) });
        this.active = job.id;
        const cancel = () => { void this.cancel().catch(this.progress); };
        signal?.addEventListener('abort', cancel, { once: true });
        try {
            if (signal?.aborted)
                await this.cancel();
            for (;;) {
                const record = await this.request(`jobs/${job.id}`);
                this.progress(`${operation}: ${record.status}`, record);
                if (record.status === 'complete')
                    return record.result;
                if (record.status !== 'running')
                    throw new Error(record.error ?? `Operation ${record.status}`);
                await new Promise(resolve => setTimeout(resolve, 300));
            }
        }
        finally {
            signal?.removeEventListener('abort', cancel);
            this.active = null;
        }
    }
    async cancel() { if (this.active)
        await this.request(`jobs/${this.active}/cancel`, { method: 'POST' }); }
    async upload(file) { return this.request('upload', { method: 'POST', body: file }); }
    assetUrl(asset) { return `/studio-assets/${this.session}/${asset}`; }
}
