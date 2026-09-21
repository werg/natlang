/** Versioned local artifacts for natlang-authored programs and investigations.
 * Artifact content is immutable. Only the active manifest pointer changes.
 */
const copy = value => structuredClone(value);
const pathPattern = /^[A-Za-z0-9][A-Za-z0-9_./-]{0,239}$/;
function canonical(value) {
    if (Array.isArray(value)) return value.map(canonical);
    if (value && typeof value === 'object')
        return Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])]));
    return value;
}
async function digest(value) {
    const bytes = new TextEncoder().encode(JSON.stringify(canonical(value)));
    const hash = await crypto.subtle.digest('SHA-256', bytes);
    return [...new Uint8Array(hash)].map(x => x.toString(16).padStart(2, '0')).join('');
}
function assert(test, message) { if (!test) throw new Error(message); }
function validatePath(path) {
    assert(typeof path === 'string' && pathPattern.test(path) && !path.split('/').includes('..'), `Invalid artifact path: ${path}`);
}
function validateArtifact(artifact) {
    assert(artifact && ['source', 'schema', 'view', 'data', 'evidence', 'claim', 'intent', 'assessment', 'migration', 'method', 'change'].includes(artifact.kind), 'Unknown artifact kind');
    assert(Object.hasOwn(artifact, 'content'), 'Artifact has no content');
    assert(JSON.stringify(artifact).length <= 2_000_000, 'Store large native data by reference');
}

/** A small adapter is sufficient for IndexedDB and deterministic tests. */
export class ResearchWorkspace {
    constructor(adapter, key = 'research') { this.adapter = adapter; this.key = key; }
    static empty() { return { format: 1, head: '', manifests: {}, artifacts: {} }; }
    async snapshot() { return copy(await this.adapter.readWorkspace(this.key) ?? ResearchWorkspace.empty()); }
    async head() { const snapshot = await this.snapshot(); return copy(snapshot.manifests[snapshot.head] ?? null); }
    async artifact(id) {
        const snapshot = await this.snapshot();
        const artifact = snapshot.artifacts[id];
        assert(artifact, `Missing artifact ${id}`);
        assert(await digest(artifact) === id, `Corrupt artifact ${id}`);
        return copy(artifact);
    }
    async at(manifestId, path) {
        validatePath(path);
        const snapshot = await this.snapshot();
        const manifest = snapshot.manifests[manifestId];
        assert(manifest, `Unknown manifest ${manifestId}`);
        const id = manifest.files[path];
        assert(id, `Unknown artifact path ${path}`);
        const artifact = snapshot.artifacts[id];
        assert(artifact && await digest(artifact) === id, `Corrupt artifact ${path}`);
        return copy({ id, ...artifact });
    }
    async stage(baseId, edits, { effects = [], message = '' } = {}) {
        const snapshot = await this.snapshot();
        assert(baseId === '' || snapshot.manifests[baseId], `Unknown base manifest ${baseId}`);
        assert(edits && typeof edits === 'object' && !Array.isArray(edits), 'Edits must be a path map');
        const files = { ...(snapshot.manifests[baseId]?.files ?? {}) };
        const newArtifacts = {};
        for (const [path, artifact] of Object.entries(edits)) {
            validatePath(path);
            if (artifact === null) { delete files[path]; continue; }
            validateArtifact(artifact);
            const clean = copy(artifact);
            const id = await digest(clean);
            files[path] = id;
            newArtifacts[id] = clean;
        }
        for (const id of effects) assert(typeof id === 'string' && id, 'Effect reference needs an ID');
        const body = { parent: baseId, files, effects: [...new Set(effects)], message: String(message) };
        const id = await digest(body);
        return { manifest: { id, ...body }, artifacts: newArtifacts };
    }
    async commit(expectedHead, staged) {
        const before = await this.snapshot();
        assert(before.head === expectedHead, 'Workspace changed; rebase the candidate');
        assert(staged.manifest.parent === expectedHead, 'Candidate has a different base');
        const next = copy(before);
        for (const [id, artifact] of Object.entries(staged.artifacts)) {
            validateArtifact(artifact);
            assert(await digest(artifact) === id, 'Artifact digest mismatch');
            next.artifacts[id] = copy(artifact);
        }
        const { manifest } = staged;
        const { id, ...body } = manifest;
        assert(await digest(body) === id, 'Manifest digest mismatch');
        for (const artifactId of Object.values(manifest.files))
            assert(next.artifacts[artifactId], `Manifest references missing artifact ${artifactId}`);
        for (const effectId of manifest.effects)
            assert(await this.adapter.hasEffect(effectId), `Manifest references missing effect ${effectId}`);
        next.manifests[id] = copy(manifest);
        next.head = id;
        await this.adapter.compareAndSwapWorkspace(this.key, expectedHead, next);
        return copy(manifest);
    }
    async commitEdits(expectedHead, edits, options) {
        return this.commit(expectedHead, await this.stage(expectedHead, edits, options));
    }
    async branch(baseId, edits, options) { return this.stage(baseId, edits, options); }
    /** Persist a branch without moving the active pointer. */
    async propose(baseId, edits, options) {
        const staged = await this.stage(baseId, edits, options);
        const before = await this.snapshot(), next = copy(before);
        for (const [id, artifact] of Object.entries(staged.artifacts)) next.artifacts[id] = copy(artifact);
        for (const effectId of staged.manifest.effects)
            assert(await this.adapter.hasEffect(effectId), `Candidate references missing effect ${effectId}`);
        next.manifests[staged.manifest.id] = copy(staged.manifest);
        await this.adapter.compareAndSwapWorkspace(this.key, before.head, next);
        return copy(staged.manifest);
    }
    async activate(expectedHead, candidateId) {
        const before = await this.snapshot(), candidate = before.manifests[candidateId];
        assert(candidate, `Unknown candidate ${candidateId}`);
        assert(candidate.parent === expectedHead && before.head === expectedHead, 'Candidate needs semantic rebase before activation');
        const next = copy(before); next.head = candidateId;
        await this.adapter.compareAndSwapWorkspace(this.key, expectedHead, next);
        return copy(candidate);
    }
    async export() {
        const bundle = await this.snapshot();
        const effects = new Set(Object.values(bundle.manifests).flatMap(manifest => manifest.effects));
        bundle.receipts = {};
        for (const id of effects) {
            const receipt = await this.adapter.readEffect(id);
            assert(receipt, `Missing effect receipt ${id}`);
            bundle.receipts[id] = receipt;
        }
        return bundle;
    }
    async import(bundle, expectedHead = '') {
        assert(bundle?.format === 1 && bundle.manifests && bundle.artifacts, 'Unknown workspace bundle');
        assert(bundle.head === '' || bundle.manifests[bundle.head], 'Bundle head is missing');
        for (const [id, artifact] of Object.entries(bundle.artifacts)) {
            validateArtifact(artifact);
            assert(await digest(artifact) === id, `Corrupt imported artifact ${id}`);
        }
        for (const [id, manifest] of Object.entries(bundle.manifests)) {
            const { id: embedded, ...body } = manifest;
            assert(embedded === id && await digest(body) === id, `Corrupt imported manifest ${id}`);
            assert(!manifest.parent || bundle.manifests[manifest.parent], `Missing parent ${manifest.parent}`);
            for (const [path, artifactId] of Object.entries(manifest.files)) {
                validatePath(path);
                assert(bundle.artifacts[artifactId], `Missing imported artifact ${artifactId}`);
            }
            for (const effectId of manifest.effects)
                assert(bundle.receipts?.[effectId]?.id === effectId, `Missing imported receipt ${effectId}`);
        }
        for (const receipt of Object.values(bundle.receipts ?? {}))
            await this.adapter.recordEffect({ ...receipt, imported: true });
        const { receipts, ...snapshot } = bundle;
        await this.adapter.compareAndSwapWorkspace(this.key, expectedHead, copy(snapshot));
    }
}

export class MemoryResearchAdapter {
    constructor() { this.workspaces = new Map(); this.effects = new Map(); }
    async readWorkspace(key) { return copy(this.workspaces.get(key)); }
    async compareAndSwapWorkspace(key, expectedHead, next) {
        const previous = this.workspaces.get(key);
        assert((previous?.head ?? '') === expectedHead, 'Workspace changed; rebase the candidate');
        this.workspaces.set(key, copy(next));
    }
    async hasEffect(id) { return this.effects.has(id) && this.effects.get(id).status !== 'running'; }
    async readEffect(id) { return copy(this.effects.get(id)); }
    async beginEffect(receipt) {
        assert(!this.effects.has(receipt.id), 'Effect ID already exists');
        this.effects.set(receipt.id, copy({ ...receipt, status: 'running' }));
    }
    async finishEffect(id, receipt) {
        assert(this.effects.get(id)?.status === 'running', 'Effect is not pending');
        assert(receipt.id === id, 'Effect ID mismatch');
        this.effects.set(id, copy(receipt));
    }
    async recordEffect(receipt) {
        assert(receipt?.id && receipt.status, 'Effect receipt needs id and status');
        const prior = this.effects.get(receipt.id);
        assert(!prior || JSON.stringify(prior) === JSON.stringify(receipt), 'Effect ID has a different outcome');
        this.effects.set(receipt.id, copy(receipt));
    }
}
