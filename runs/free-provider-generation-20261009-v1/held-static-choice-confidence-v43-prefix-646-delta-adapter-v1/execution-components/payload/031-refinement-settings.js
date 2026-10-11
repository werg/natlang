/** The `refinements` settings of natlang.json (plans/REFINEMENT_TYPES.md section 3), parsed and validated. */
// --- Settings (natlang.json `refinements`) -------------------------------------------------------------------------
const unit = (value, label) => {
    if (typeof value !== 'number' || !(value >= 0 && value <= 1))
        throw new TypeError(`${label} must be a number from 0 to 1`);
    return value;
};
function parsePredicateSettings(raw, label, extra) {
    const known = new Set(['threshold', 'band', 'policy', 'mode', ...extra]);
    for (const key of Object.keys(raw))
        if (!known.has(key))
            throw new TypeError(`unknown ${label} field: ${key}`);
    const out = {};
    if (raw.threshold !== undefined)
        out.threshold = unit(raw.threshold, `${label}.threshold`);
    if (raw.band !== undefined) {
        const band = raw.band;
        if (!band || typeof band !== 'object' || Array.isArray(band))
            throw new TypeError(`${label}.band must be { low, high }`);
        const low = unit(band.low, `${label}.band.low`), high = unit(band.high, `${label}.band.high`);
        if (low > high)
            throw new TypeError(`${label}.band.low must not exceed high`);
        out.band = { low, high };
    }
    if (raw.policy !== undefined) {
        if (!['accept', 'reject', 'escalate'].includes(raw.policy))
            throw new TypeError(`${label}.policy must be "accept", "reject" or "escalate"`);
        out.policy = raw.policy;
    }
    if (raw.mode !== undefined) {
        if (!['crisp', 'nl', 'shadow'].includes(raw.mode))
            throw new TypeError(`${label}.mode must be "crisp", "nl" or "shadow"`);
        out.mode = raw.mode;
    }
    return out;
}
/** Validate the `refinements` object of natlang.json. */
export function parseRefinementSettings(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value))
        throw new TypeError('refinements must be an object');
    const raw = value;
    const out = parsePredicateSettings(raw, 'refinements', ['judge', 'escalate', 'repairs', 'predicates', 'services']);
    for (const key of ['judge', 'escalate'])
        if (raw[key] !== undefined) {
            if (typeof raw[key] !== 'string' || !raw[key])
                throw new TypeError(`refinements.${key} must be a model name`);
            out[key] = raw[key];
        }
    if (raw.repairs !== undefined) {
        if (!Number.isInteger(raw.repairs) || raw.repairs < 0)
            throw new TypeError('refinements.repairs must be a whole number');
        out.repairs = raw.repairs;
    }
    if (raw.predicates !== undefined) {
        const predicates = raw.predicates;
        if (!predicates || typeof predicates !== 'object' || Array.isArray(predicates))
            throw new TypeError('refinements.predicates must be an object');
        out.predicates = {};
        for (const [text, item] of Object.entries(predicates)) {
            if (!item || typeof item !== 'object' || Array.isArray(item))
                throw new TypeError(`refinements.predicates[${JSON.stringify(text)}] must be an object`);
            out.predicates[normalize(text)] = parsePredicateSettings(item, `refinements.predicates[${JSON.stringify(text)}]`, []);
        }
    }
    if (raw.services !== undefined) {
        const services = raw.services;
        if (!services || typeof services !== 'object' || Array.isArray(services))
            throw new TypeError('refinements.services must be an object');
        out.services = {};
        for (const [name, type] of Object.entries(services)) {
            if (typeof type !== 'string' || !type)
                throw new TypeError(`refinements.services.${name} must be type text`);
            out.services[name] = type;
        }
    }
    return out;
}
export const normalizePredicate = (predicate) => predicate.replace(/\s+/g, ' ').trim();
const normalize = normalizePredicate;
