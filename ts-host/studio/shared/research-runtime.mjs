import { ResearchWorkspace } from './research-workspace.mjs';
import { sameValue } from './domain.mjs';

function assert(test, message) { if (!test) throw new Error(message); }
function copy(value) { return structuredClone(value); }
function sourceFiles(snapshot, manifest) {
    const files = {};
    for (const [path, id] of Object.entries(manifest.files)) {
        const artifact = snapshot.artifacts[id];
        if ((artifact?.kind === 'source' || artifact?.kind === 'schema') && /\.(nl|ts)$/.test(path)) {
            assert(typeof artifact.content === 'string', `Source ${path} is not text`);
            files[path] = artifact.content;
        }
    }
    return files;
}

const tags = new Set(['main', 'section', 'div', 'h1', 'h2', 'h3', 'p', 'span', 'strong', 'em', 'ul', 'ol', 'li', 'button', 'input', 'label', 'output',
    'table', 'thead', 'tbody', 'tr', 'th', 'td', 'textarea', 'select', 'option', 'details', 'summary', 'pre', 'code', 'meter', 'progress', 'br']);
/** Exact shape and wiring validation; the model chooses the actual interaction. */
export function validateInteraction(tree, bindings) {
    assert(bindings && typeof bindings === 'object' && !Array.isArray(bindings), 'Bindings must be a record');
    const controls = new Set(), inputs = new Set();
    const seen = new WeakSet();
    const pending = [tree];
    while (pending.length) {
        const node = pending.pop();
        assert(node && typeof node === 'object' && tags.has(node.tag), `Unsupported view tag ${node?.tag}`);
        assert(!seen.has(node), 'View tree reuses a node');
        seen.add(node);
        assert(!('html' in node) && !('script' in node), 'Raw HTML and script are not view data');
        if (node.id) {
            assert(typeof node.id === 'string' && !controls.has(node.id), `Duplicate view ID ${node.id}`);
            controls.add(node.id);
        }
        if (['input', 'textarea', 'select'].includes(node.tag)) {
            assert(node.id, 'Input needs a stable ID');
            inputs.add(node.id);
        }
        if (node.action) {
            assert(node.id && ['input', 'textarea', 'select', 'button'].includes(node.tag), 'Only identified controls can emit events');
            assert(node.action.kind === node.id, 'Control action must identify its own binding');
        }
        assert(node.children === undefined || Array.isArray(node.children), 'View children must be a list');
        for (const child of node.children ?? []) pending.push(child);
    }
    for (const [id, binding] of Object.entries(bindings)) {
        assert(controls.has(id), `Binding has no control ${id}`);
        assert(binding && typeof binding.root === 'string' && binding.root.endsWith('.nl'), `Binding ${id} needs a natlang handler`);
        if (binding.from) assert(inputs.has(binding.from), `Binding ${id} has no input ${binding.from}`);
    }
    return { controls: [...controls], inputs: [...inputs] };
}

export function validateGeneratedModule(module, bindings) {
    assert(module && typeof module === 'object' && !Array.isArray(module), 'Generated module must be a record');
    assert(typeof module.html === 'string' && typeof module.script === 'string', 'Generated module needs html and script text');
    assert(module.style === undefined || typeof module.style === 'string', 'Generated module style must be text');
    assert(bindings && typeof bindings === 'object' && !Array.isArray(bindings), 'Bindings must be a record');
    for (const [id, binding] of Object.entries(bindings)) {
        assert(id && binding && typeof binding.root === 'string' && (binding.root.endsWith('.nl') || binding.root.endsWith('.ts')),
            `Binding ${id} needs a natlang or crisp handler`);
    }
    return { controls: Object.keys(bindings) };
}

/** Application host capabilities. Meaning stays in natlang source and state. */
export class ResearchRuntime {
    constructor({ adapter, runSource, runContext = () => ({}), key = 'research' }) {
        this.adapter = adapter;
        this.workspace = new ResearchWorkspace(adapter, key);
        this.runSource = runSource;
        this.runContext = runContext;
    }
    async manifest(id) {
        const snapshot = await this.workspace.snapshot();
        const manifest = snapshot.manifests[id || snapshot.head];
        assert(manifest, `Unknown manifest ${id}`);
        return copy(manifest);
    }
    async read(manifestId, path) { return this.workspace.at(manifestId, path); }
    async list(manifestId, kinds) {
        const snapshot = await this.workspace.snapshot(), manifest = snapshot.manifests[manifestId];
        assert(manifest, `Unknown manifest ${manifestId}`);
        return Object.entries(manifest.files).filter(([, id]) => !kinds?.length || kinds.includes(snapshot.artifacts[id].kind))
            .map(([path, id]) => ({ path, id, kind: snapshot.artifacts[id].kind }));
    }
    async search(manifestId, text, kinds) {
        assert(typeof text === 'string' && text.trim(), 'Search text is required');
        const snapshot = await this.workspace.snapshot(), manifest = snapshot.manifests[manifestId];
        assert(manifest, `Unknown manifest ${manifestId}`);
        const needle = text.toLowerCase();
        return Object.entries(manifest.files).flatMap(([path, id]) => {
            const artifact = snapshot.artifacts[id];
            if (kinds?.length && !kinds.includes(artifact.kind)) return [];
            const value = typeof artifact.content === 'string' ? artifact.content : JSON.stringify(artifact.content);
            const index = value.toLowerCase().indexOf(needle);
            return index < 0 ? [] : [{ path, id, kind: artifact.kind, offset: index,
                excerpt: value.slice(Math.max(0, index - 100), index + needle.length + 100) }];
        });
    }
    async execute(manifestId, root, inputs, callId) {
        assert(typeof callId === 'string' && callId, 'Execution needs a stable call ID');
        const prior = await this.adapter.readEffect(callId);
        if (prior) {
            assert(prior.manifest === manifestId && prior.root === root && sameValue(prior.inputs, inputs), 'Call ID was reused for different inputs');
            if (prior.status === 'running') {
                const unknown = { ...prior, status: 'unknown', error: 'Execution was interrupted; inspect external effects before retrying with a new ID' };
                await this.adapter.finishEffect(callId, unknown);
                return copy(unknown);
            }
            return copy(prior);
        }
        const snapshot = await this.workspace.snapshot(), manifest = snapshot.manifests[manifestId];
        assert(manifest, `Unknown manifest ${manifestId}`);
        const files = sourceFiles(snapshot, manifest);
        assert(Object.hasOwn(files, root), `Missing root source ${root}`);
        const provenance = copy(this.runContext());
        await this.adapter.beginEffect({ id: callId, manifest: manifestId, root, inputs: copy(inputs), provenance });
        // A loaded model and selected eval environment are supplied by the embedding.
        // This boundary stores the actual result, including failure, as an effect.
        let receipt;
        try {
            const native_ids = [...new Set(Object.values(manifest.files).map(id => snapshot.artifacts[id]?.content?.native_id).filter(Boolean))];
            const result = await this.runSource(Object.entries(files).map(([id, source]) => ({ id, source })), root, copy(inputs), { native_ids, provenance });
            receipt = { id: callId, status: 'complete', manifest: manifestId, root, inputs: copy(inputs),
                value: copy(result.value), trace_id: result.trace_id ?? '', provenance };
        }
        catch (error) {
            const detail = String(error);
            receipt = { id: callId, status: /cancel|abort|terminat/i.test(detail) ? 'unknown' : 'failed',
                manifest: manifestId, root, inputs: copy(inputs), error: detail, provenance };
        }
        await this.adapter.finishEffect(callId, receipt);
        return copy(receipt);
    }
    async commit(base, edits, options) { return this.workspace.commitEdits(base, edits, options); }
    async stage(base, edits, options) { return this.workspace.stage(base, edits, options); }
    async propose(base, edits, options) { return this.workspace.propose(base, edits, options); }
    async activate(base, candidateId) { return this.workspace.activate(base, candidateId); }
    async diff(leftId, rightId) {
        const snapshot = await this.workspace.snapshot(), left = snapshot.manifests[leftId], right = snapshot.manifests[rightId];
        assert(left && right, 'Both manifests must exist');
        const paths = new Set([...Object.keys(left.files), ...Object.keys(right.files)]);
        return [...paths].sort().flatMap(path => left.files[path] === right.files[path] ? [] : [{ path,
            before: left.files[path] ?? '', after: right.files[path] ?? '' }]);
    }
    /** Exact material for human and natlang review; overlapping edits require a semantic decision. */
    async reviewCandidate(candidateId) {
        const snapshot = await this.workspace.snapshot();
        const candidate = snapshot.manifests[candidateId], active = snapshot.manifests[snapshot.head];
        assert(candidate && active && candidate.parent, 'Candidate and active manifests must exist');
        const base = snapshot.manifests[candidate.parent];
        assert(base, 'Candidate base is missing');
        const candidateChanges = await this.diff(base.id, candidate.id);
        const activeChanges = await this.diff(base.id, active.id);
        const activePaths = new Set(activeChanges.map(row => row.path));
        return {
            candidate: candidate.id, base: base.id, active: active.id,
            message: candidate.message, can_activate: candidate.parent === active.id,
            overlapping_paths: candidateChanges.filter(row => activePaths.has(row.path)).map(row => row.path),
            changes: candidateChanges.map(row => ({ path: row.path,
                before: row.before ? copy(snapshot.artifacts[row.before]) : null,
                proposed: row.after ? copy(snapshot.artifacts[row.after]) : null,
                active: active.files[row.path] ? copy(snapshot.artifacts[active.files[row.path]]) : null })),
        };
    }
    async branches() {
        const snapshot = await this.workspace.snapshot();
        const lineage = new Set();
        for (let id = snapshot.head; id && snapshot.manifests[id]; id = snapshot.manifests[id].parent) lineage.add(id);
        return Object.values(snapshot.manifests).filter(manifest => manifest.id !== snapshot.head).map(manifest => {
            const parent = snapshot.manifests[manifest.parent];
            const paths = new Set([...Object.keys(parent?.files ?? {}), ...Object.keys(manifest.files)]);
            return { id: manifest.id, parent: manifest.parent, message: manifest.message,
                changed: [...paths].filter(path => parent?.files[path] !== manifest.files[path]).length,
                active: false, kind: lineage.has(manifest.id) ? 'history' : 'candidate' };
        });
    }
    /** Exact dependency projection; natlang still judges relevance and entailment. */
    async beliefGraph(manifestId) {
        const snapshot = await this.workspace.snapshot(), manifest = snapshot.manifests[manifestId];
        assert(manifest, `Unknown manifest ${manifestId}`);
        const nodes = [], links = [], missing = [], invalid = [];
        for (const [path, id] of Object.entries(manifest.files)) {
            const artifact = snapshot.artifacts[id];
            if (!['claim', 'evidence', 'assessment'].includes(artifact.kind)) continue;
            const content = artifact.content;
            nodes.push({ path, id, kind: artifact.kind, title: typeof content === 'string' ? content.slice(0, 180) :
                String(content?.text ?? content?.claim ?? content?.summary ?? path),
                status: typeof content === 'object' ? String(content?.status ?? '') : '' });
            if (!content || typeof content !== 'object' || Array.isArray(content)) continue;
            for (const relation of ['supports', 'opposes', 'assumptions', 'questions', 'depends_on']) {
                const targets = content[relation] ?? [];
                if (!Array.isArray(targets)) { invalid.push({ path, relation, reason: 'Expected a list of artifact paths' }); continue; }
                for (const target of targets) {
                    if (typeof target !== 'string' || !target) {
                        invalid.push({ path, relation, reason: 'Expected a nonempty artifact path' }); continue;
                    }
                    const edge = { from: path, to: target, relation };
                    (manifest.files[target] ? links : missing).push(edge);
                }
            }
        }
        return { nodes, links, missing, invalid };
    }
    async affected(manifestId, changedPaths) {
        const { links } = await this.beliefGraph(manifestId);
        const affected = new Set(changedPaths);
        for (let i = 0; i < links.length + 1; i++) {
            let grew = false;
            for (const link of links) if (affected.has(link.to) && !affected.has(link.from)) {
                affected.add(link.from); grew = true;
            }
            if (!grew) break;
        }
        return [...affected].filter(path => !changedPaths.includes(path)).sort();
    }
    /** Compare a real migration result with immutable input records. The mapping keys
     * are supplied by the authored migration; this reports mechanics, not meaning. */
    async auditMigration(manifestId, sourcePath, receiptId, spec) {
        const source = await this.read(manifestId, sourcePath);
        const receipt = await this.adapter.readEffect(receiptId);
        assert(receipt?.status === 'complete' && receipt.manifest === manifestId,
            'Migration audit needs a completed receipt from this manifest');
        const input = source.content, output = receipt.value;
        assert(Array.isArray(input) && Array.isArray(output), 'Migration input and output must be record lists');
        assert(spec && typeof spec === 'object' && !Array.isArray(spec), 'Migration audit needs a mapping specification');
        const sourceKey = spec.source_key, outputKey = spec.output_source_key;
        const preserved = spec.preserved_fields ?? [];
        assert(typeof sourceKey === 'string' && sourceKey && typeof outputKey === 'string' && outputKey,
            'Migration audit needs source_key and output_source_key');
        assert(Array.isArray(preserved) && preserved.every(field => typeof field === 'string' && field),
            'preserved_fields must be a list of field names');
        const bySource = new Map(), duplicateInputs = [];
        for (const record of input) {
            assert(record && typeof record === 'object' && !Array.isArray(record), 'Migration input contains a non-record');
            const key = record[sourceKey];
            assert(['string', 'number'].includes(typeof key), `Input record lacks scalar ${sourceKey}`);
            if (bySource.has(key)) duplicateInputs.push(key); else bySource.set(key, record);
        }
        const counts = new Map(), unknown = [], fieldMismatches = [];
        for (const record of output) {
            assert(record && typeof record === 'object' && !Array.isArray(record), 'Migration output contains a non-record');
            const key = record[outputKey];
            if (!bySource.has(key)) { unknown.push(key); continue; }
            counts.set(key, (counts.get(key) ?? 0) + 1);
            const original = bySource.get(key);
            for (const field of preserved) if (!sameValue(original[field], record[field]))
                fieldMismatches.push({ key, field, before: copy(original[field]), after: copy(record[field]) });
        }
        const unmatched = [...bySource.keys()].filter(key => !counts.has(key));
        const multiplied = [...counts].filter(([, count]) => count > 1).map(([key, count]) => ({ key, count }));
        return { input_count: input.length, output_count: output.length,
            matched_inputs: bySource.size - unmatched.length, unmatched, multiplied,
            unknown_output_sources: unknown, duplicate_input_keys: duplicateInputs,
            preserved_field_mismatches: fieldMismatches };
    }
    async interaction(manifestId, path) {
        const artifact = await this.read(manifestId, path);
        assert(artifact.kind === 'view', 'Interaction path must contain a view');
        const { tree, module, bindings } = artifact.content;
        assert(Boolean(tree) !== Boolean(module), 'Interaction needs exactly one tree or module');
        if (tree) validateInteraction(tree, bindings); else validateGeneratedModule(module, bindings);
        const snapshot = await this.workspace.snapshot(), manifest = snapshot.manifests[manifestId];
        for (const binding of Object.values(bindings))
            assert(sourceFiles(snapshot, manifest)[binding.root], `Missing interaction handler ${binding.root}`);
        return copy({ revision: manifestId, ...(tree ? { tree } : { module }), bindings });
    }
}
