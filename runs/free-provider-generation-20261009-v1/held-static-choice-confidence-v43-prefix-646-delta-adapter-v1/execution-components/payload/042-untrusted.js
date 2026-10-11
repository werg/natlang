const MAX_ENTRIES = 20_000;
const MAX_CHARS = 8_000_000;
export class UntrustedRegistry {
    parent;
    sources = new Map();
    chars = 0;
    /** A registry also asks `parent`: host code marks text with `untrusted(value, source)` before any task exists. */
    constructor(parent) {
        this.parent = parent;
    }
    /** The source that produced `text`, or undefined when it is not untrusted. */
    sourceOf(text) { return text ? this.sources.get(text) ?? this.parent?.sourceOf(text) : undefined; }
    /** Record every string inside `value` as coming from `source`. The first source of a text is kept. */
    markAll(value, source, depth = 0) {
        if (typeof value === 'string') {
            this.add(value, source);
            return;
        }
        if (depth > 48 || !value || typeof value !== 'object')
            return;
        if (Array.isArray(value)) {
            for (const item of value)
                this.markAll(item, source, depth + 1);
            return;
        }
        if (Object.prototype.toString.call(value) !== '[object Object]')
            return;
        for (const item of Object.values(value))
            this.markAll(item, source, depth + 1);
    }
    /** Record the strings of `value` that sit at `Untrusted<...>` positions of `type`. */
    markTyped(value, type, env, source) {
        const walk = (item, current, depth) => {
            if (depth > 32 || item === undefined || item === null)
                return;
            let resolved;
            try {
                resolved = env.resolve(current);
            }
            catch {
                return;
            }
            switch (resolved.kind) {
                case 'untrusted':
                    this.markAll(item, source);
                    return;
                case 'refined':
                    walk(item, resolved.base, depth + 1);
                    return;
                case 'union':
                    for (const member of resolved.members)
                        walk(item, member, depth + 1);
                    return;
                case 'list':
                    if (Array.isArray(item))
                        for (const child of item)
                            walk(child, resolved.element, depth + 1);
                    return;
                case 'dict':
                    if (typeof item === 'object')
                        for (const child of Object.values(item))
                            walk(child, resolved.element, depth + 1);
                    return;
                case 'record':
                    if (typeof item === 'object' && !Array.isArray(item))
                        for (const field of resolved.fields)
                            walk(item[field.name], field.type, depth + 1);
                    return;
                default: return;
            }
        };
        walk(value, type, 0);
    }
    add(text, source) {
        if (!text || this.sourceOf(text) !== undefined)
            return;
        this.sources.set(text, source);
        this.chars += text.length;
        // Oldest first: a long run does not grow the registry without bound.
        for (const key of this.sources.keys()) {
            if (this.sources.size <= MAX_ENTRIES && this.chars <= MAX_CHARS)
                break;
            this.sources.delete(key);
            this.chars -= key.length;
        }
    }
}
/** Text the host marked itself with `untrusted(value, source)`, outside any task. */
export const hostUntrusted = new UntrustedRegistry();
const scoped = new WeakMap();
/** The registry of one execution scope (a natlang task): calls of that task share what they have seen. */
export function scopedUntrusted(scope) {
    let registry = scoped.get(scope);
    if (!registry) {
        registry = new UntrustedRegistry(hostUntrusted);
        scoped.set(scope, registry);
    }
    return registry;
}
/** Whether `type` has an `Untrusted<...>` position anywhere (named types followed once each). */
export function containsUntrusted(type, env, seen = new Set()) {
    if (type.kind === 'untrusted')
        return true;
    if (type.kind === 'name') {
        if (seen.has(type.name))
            return false;
        seen.add(type.name);
        const found = env.lookup(type.name);
        return !!found && containsUntrusted(found, env, seen);
    }
    switch (type.kind) {
        case 'record': return type.fields.some(field => containsUntrusted(field.type, env, seen));
        case 'list':
        case 'dict': return containsUntrusted(type.element, env, seen);
        case 'refined': return containsUntrusted(type.base, env, seen);
        case 'union': return type.members.some(member => containsUntrusted(member, env, seen));
        default: return false;
    }
}
/**
 * An untrusted string as a quoted data block: a fence longer than any run of backticks in the text, so the text cannot
 * close it, and an info line naming the source. The text is verbatim.
 */
export function untrustedBlock(text, source) {
    const longest = Math.max(0, ...(text.match(/`+/g) ?? []).map(run => run.length));
    const fence = '`'.repeat(Math.max(3, longest + 1));
    const label = source.replace(/[^\x20-\x7e]|`/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 80) || 'outside the program';
    return `\n${fence}untrusted data from ${label}\n${text}\n${fence}\n`;
}
