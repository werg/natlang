import { ResearchWorkspace } from './research-workspace.mjs';

function assert(test, message) { if (!test) throw new Error(message); }
function copy(value) { return structuredClone(value); }
function sourceFiles(snapshot, manifest) {
    const files = {};
    for (const [path, id] of Object.entries(manifest.files)) {
        const artifact = snapshot.artifacts[id];
        if (artifact?.kind === 'source' || artifact?.kind === 'schema') {
            assert(typeof artifact.content === 'string', `Source ${path} is not text`);
            files[path] = artifact.content;
        }
    }
    return files;
}

const tags = new Set(['main', 'section', 'div', 'h1', 'h2', 'h3', 'p', 'span', 'strong', 'em', 'ul', 'ol', 'li', 'button', 'input', 'label', 'output']);
/** Exact shape and wiring validation; the model chooses the actual interaction. */
export function validateInteraction(tree, bindings) {
    assert(bindings && typeof bindings === 'object' && !Array.isArray(bindings), 'Bindings must be a record');
    const controls = new Set(), inputs = new Set();
    let count = 0;
    function visit(node, depth) {
        assert(node && typeof node === 'object' && tags.has(node.tag), `Unsupported view tag ${node?.tag}`);
        assert(++count <= 2000 && depth <= 32, 'View exceeds renderer bounds');
        assert(!('html' in node) && !('script' in node), 'Raw HTML and script are not view data');
        if (node.id) {
            assert(typeof node.id === 'string' && !controls.has(node.id), `Duplicate view ID ${node.id}`);
            controls.add(node.id);
        }
        if (node.tag === 'input') {
            assert(node.id, 'Input needs a stable ID');
            inputs.add(node.id);
        }
        if (node.action) {
            assert(node.id && (node.tag === 'input' || node.tag === 'button'), 'Only identified controls can emit events');
            assert(node.action.kind === node.id, 'Control action must identify its own binding');
        }
        for (const child of node.children ?? []) visit(child, depth + 1);
    }
    visit(tree, 0);
    for (const [id, binding] of Object.entries(bindings)) {
        assert(controls.has(id), `Binding has no control ${id}`);
        assert(binding && typeof binding.root === 'string' && binding.root.endsWith('.nl'), `Binding ${id} needs a natlang handler`);
        if (binding.from) assert(inputs.has(binding.from), `Binding ${id} has no input ${binding.from}`);
    }
    return { controls: [...controls], inputs: [...inputs] };
}

/** Application host capabilities. Meaning stays in natlang source and state. */
export class ResearchRuntime {
    constructor({ adapter, runSource, key = 'research' }) {
        this.adapter = adapter;
        this.workspace = new ResearchWorkspace(adapter, key);
        this.runSource = runSource;
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
            assert(prior.manifest === manifestId && prior.root === root && JSON.stringify(prior.inputs) === JSON.stringify(inputs), 'Call ID was reused for different inputs');
            return copy(prior);
        }
        const snapshot = await this.workspace.snapshot(), manifest = snapshot.manifests[manifestId];
        assert(manifest, `Unknown manifest ${manifestId}`);
        const files = sourceFiles(snapshot, manifest);
        assert(Object.hasOwn(files, root), `Missing root source ${root}`);
        // A loaded model and selected eval environment are supplied by the embedding.
        // This boundary stores the actual result, including failure, as an effect.
        let receipt;
        try {
            const result = await this.runSource(Object.entries(files).map(([id, source]) => ({ id, source })), root, copy(inputs));
            receipt = { id: callId, status: 'complete', manifest: manifestId, root, inputs: copy(inputs),
                value: copy(result.value), trace_id: result.trace_id ?? '' };
        }
        catch (error) {
            receipt = { id: callId, status: 'failed', manifest: manifestId, root, inputs: copy(inputs), error: String(error) };
        }
        await this.adapter.recordEffect(receipt);
        return copy(receipt);
    }
    async commit(base, edits, options) { return this.workspace.commitEdits(base, edits, options); }
    async stage(base, edits, options) { return this.workspace.stage(base, edits, options); }
    async activate(base, candidate) { return this.workspace.commit(base, candidate); }
    async diff(leftId, rightId) {
        const snapshot = await this.workspace.snapshot(), left = snapshot.manifests[leftId], right = snapshot.manifests[rightId];
        assert(left && right, 'Both manifests must exist');
        const paths = new Set([...Object.keys(left.files), ...Object.keys(right.files)]);
        return [...paths].sort().flatMap(path => left.files[path] === right.files[path] ? [] : [{ path,
            before: left.files[path] ?? '', after: right.files[path] ?? '' }]);
    }
    async interaction(manifestId, path) {
        const artifact = await this.read(manifestId, path);
        assert(artifact.kind === 'view', 'Interaction path must contain a view');
        const { tree, bindings } = artifact.content;
        validateInteraction(tree, bindings);
        const snapshot = await this.workspace.snapshot(), manifest = snapshot.manifests[manifestId];
        for (const binding of Object.values(bindings))
            assert(sourceFiles(snapshot, manifest)[binding.root], `Missing interaction handler ${binding.root}`);
        return copy({ revision: manifestId, tree, bindings });
    }
}
