import { ResearchRuntime } from '../shared/research-runtime.mjs';
import { valuePreview } from '../shared/domain.mjs';

/** Bind the research program to one workspace and one in-flight event. */
export class ResearchHost {
    constructor({ store, runSource, runContext }) {
        this.store = store;
        this.runtime = new ResearchRuntime({ adapter: store, runSource, runContext });
        this.event = null;
        this.effects = [];
    }
    begin(event) { this.event = event; this.effects = []; }
    end() { this.event = null; }
    editsMap(edits, removes) {
        const next = {};
        for (const edit of edits) {
            if (Object.hasOwn(next, edit.path)) throw new Error('Duplicate artifact edit');
            let content = edit.content;
            if (edit.kind === 'view') content = JSON.parse(content);
            else if (!['source', 'schema'].includes(edit.kind)) {
                try { content = JSON.parse(content); } catch { /* prose is a valid artifact */ }
            }
            next[edit.path] = { kind: edit.kind, content };
        }
        for (const path of removes) {
            if (Object.hasOwn(next, path)) throw new Error('Edit and removal target the same path');
            next[path] = null;
        }
        return next;
    }
    api() { return {
        list: head => this.runtime.list(head),
        search: (head, query) => this.runtime.search(head, query),
        read: async (head, path) => JSON.stringify(await this.runtime.read(head, path)),
        nativeRead: async (id, offset, length) => {
            const record = await this.store.get('native_values', id);
            if (!record || typeof record.value !== 'string') throw new Error('Unknown native evidence');
            if (!Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(length) || length < 1 || length > 65536)
                throw new Error('Read a valid window of at most 65536 characters');
            return record.value.slice(offset, offset + length);
        },
        nativeSearch: async (id, query) => {
            const record = await this.store.get('native_values', id);
            if (!record || typeof record.value !== 'string') throw new Error('Unknown native evidence');
            if (!query || typeof query !== 'string') throw new Error('Search query is required');
            const text = record.value, needle = query.toLowerCase(), folded = text.toLowerCase(), hits = [];
            let from = 0, total = 0;
            while (true) {
                const at = folded.indexOf(needle, from);
                if (at < 0) break;
                total++;
                if (hits.length < 30) hits.push({ offset: at, excerpt: text.slice(Math.max(0, at - 100), at + query.length + 100) });
                from = at + Math.max(1, query.length);
            }
            return JSON.stringify({ hits, total, truncated: total > hits.length, length: text.length });
        },
        diff: async (left, right) => JSON.stringify(await this.runtime.diff(left, right)),
        reviewCandidate: async id => JSON.stringify(await this.runtime.reviewCandidate(id)),
        branches: () => this.runtime.branches(),
        beliefGraph: async head => JSON.stringify(await this.runtime.beliefGraph(head)),
        affected: async (head, changed) => this.runtime.affected(head, changed),
        auditMigration: async (head, source, receipt, spec) => JSON.stringify(
            await this.runtime.auditMigration(head, source, receipt, JSON.parse(spec))),
        receipt: async id => {
            const receipt = await this.store.readEffect(id);
            if (!receipt?.root || !receipt.manifest) throw new Error('Unknown research receipt');
            return JSON.stringify(receipt);
        },
        commit: async (head, edits, removes, effectIds, message) => {
            if (!this.event) throw new Error('No active research event');
            const next = this.editsMap(edits, removes);
            const result = await this.runtime.commit(head, next, { effects: effectIds, message });
            return { head: result.id, detail: `Committed ${Object.keys(next).length} artifact changes` };
        },
        propose: async (base, edits, removes, effectIds, message) => {
            if (!this.event) throw new Error('No active research event');
            const next = this.editsMap(edits, removes);
            const candidate = await this.runtime.propose(base, next, { effects: effectIds, message });
            return { id: candidate.id, detail: `Saved candidate from ${base.slice(0, 12)}` };
        },
        activate: async (head, candidate) => {
            if (!this.event) throw new Error('No active research event');
            const result = await this.runtime.activate(head, candidate);
            return { head: result.id, detail: 'Activated reviewed candidate' };
        },
        execute: async (head, root, inputText, localCallId) => {
            if (!this.event) throw new Error('No active research event');
            if (!/^[A-Za-z0-9_.-]{1,80}$/.test(localCallId)) throw new Error('Use a stable short call ID');
            const id = `${this.event.id}/${localCallId}`;
            const receipt = await this.runtime.execute(head, root, JSON.parse(inputText), id);
            if (!this.effects.includes(id)) this.effects.push(id);
            return { id, status: receipt.status,
                value: receipt.status === 'complete' ? JSON.stringify(valuePreview(receipt.value)) : '',
                detail: receipt.status === 'complete' ? 'Actual child result recorded' : receipt.error };
        },
    }; }
    async verifyCommit(previous, next) {
        if (next.revision !== previous.revision + 1)
            throw new Error('Research revision must advance once per event');
        const head = await this.runtime.workspace.head();
        if (next.head !== (head?.id ?? '')) throw new Error('State head disagrees with committed workspace');
        if (!Array.isArray(next.receipts) || this.effects.some(id => !next.receipts.includes(id)))
            throw new Error('Reduction omitted an observed execution receipt');
        if (previous.receipts.some(id => !next.receipts.includes(id)))
            throw new Error('Reduction discarded an earlier execution receipt');
        for (const id of next.receipts) {
            const receipt = await this.store.readEffect(id);
            if (!receipt?.root || !receipt.manifest || receipt.status === 'running')
                throw new Error(`Unknown execution receipt ${id}`);
        }
        if (next.active_view) await this.runtime.interaction(next.head, next.active_view);
    }
}
