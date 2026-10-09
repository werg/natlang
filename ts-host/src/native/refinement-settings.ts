/** The `refinements` settings of natlang.json (plans/REFINEMENT_TYPES.md section 3), parsed and validated. */
// --- Settings (natlang.json `refinements`) -------------------------------------------------------------------------

/** What the checker does with a probability inside the uncertainty band. */
export type BandPolicy = 'accept' | 'reject' | 'escalate';
/** `crisp`: a registered crisp checker decides when it returns a boolean, else the judge. `nl`: the judge only.
 * `shadow`: the judge decides and every crisp/judge disagreement is recorded to the trace. */
export type RefinementMode = 'crisp' | 'nl' | 'shadow';

export type PredicateSettings = {
  /** Accept at P(true) >= threshold (default 0.5). */
  threshold?: number;
  /** Probabilities inside [low, high] are uncertain; `policy` decides them. */
  band?: { low: number; high: number };
  policy?: BandPolicy;
  mode?: RefinementMode;
};
export type RefinementSettings = PredicateSettings & {
  /** Model name (`models`) that judges; the call's own model when absent. */
  judge?: string;
  /** Model name an uncertain verdict escalates to under the `escalate` policy. */
  escalate?: string;
  /** Rejected return values the executor may repair before the call fails (default 3, or the model's maxFailureRepairs). */
  repairs?: number;
  /** Settings for one predicate (its normalized text), over the global ones. */
  predicates?: Record<string, PredicateSettings>;
  /** Declared result types of services, by `service.method`: natlang type text such as `Is<string, "a valid slug">`. */
  services?: Record<string, string>;
};

const unit = (value: unknown, label: string): number => {
  if (typeof value !== 'number' || !(value >= 0 && value <= 1)) throw new TypeError(`${label} must be a number from 0 to 1`);
  return value;
};

function parsePredicateSettings(raw: Record<string, unknown>, label: string, extra: string[]): PredicateSettings {
  const known = new Set(['threshold', 'band', 'policy', 'mode', ...extra]);
  for (const key of Object.keys(raw)) if (!known.has(key)) throw new TypeError(`unknown ${label} field: ${key}`);
  const out: PredicateSettings = {};
  if (raw.threshold !== undefined) out.threshold = unit(raw.threshold, `${label}.threshold`);
  if (raw.band !== undefined) {
    const band = raw.band as Record<string, unknown>;
    if (!band || typeof band !== 'object' || Array.isArray(band)) throw new TypeError(`${label}.band must be { low, high }`);
    const low = unit(band.low, `${label}.band.low`), high = unit(band.high, `${label}.band.high`);
    if (low > high) throw new TypeError(`${label}.band.low must not exceed high`);
    out.band = { low, high };
  }
  if (raw.policy !== undefined) {
    if (!['accept', 'reject', 'escalate'].includes(raw.policy as string)) throw new TypeError(`${label}.policy must be "accept", "reject" or "escalate"`);
    out.policy = raw.policy as BandPolicy;
  }
  if (raw.mode !== undefined) {
    if (!['crisp', 'nl', 'shadow'].includes(raw.mode as string)) throw new TypeError(`${label}.mode must be "crisp", "nl" or "shadow"`);
    out.mode = raw.mode as RefinementMode;
  }
  return out;
}

/** Validate the `refinements` object of natlang.json. */
export function parseRefinementSettings(value: unknown): RefinementSettings {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError('refinements must be an object');
  const raw = value as Record<string, unknown>;
  const out: RefinementSettings = parsePredicateSettings(raw, 'refinements', ['judge', 'escalate', 'repairs', 'predicates', 'services']);
  for (const key of ['judge', 'escalate'] as const) if (raw[key] !== undefined) {
    if (typeof raw[key] !== 'string' || !raw[key]) throw new TypeError(`refinements.${key} must be a model name`);
    out[key] = raw[key] as string;
  }
  if (raw.repairs !== undefined) {
    if (!Number.isInteger(raw.repairs) || (raw.repairs as number) < 0) throw new TypeError('refinements.repairs must be a whole number');
    out.repairs = raw.repairs as number;
  }
  if (raw.predicates !== undefined) {
    const predicates = raw.predicates as Record<string, unknown>;
    if (!predicates || typeof predicates !== 'object' || Array.isArray(predicates)) throw new TypeError('refinements.predicates must be an object');
    out.predicates = {};
    for (const [text, item] of Object.entries(predicates)) {
      if (!item || typeof item !== 'object' || Array.isArray(item)) throw new TypeError(`refinements.predicates[${JSON.stringify(text)}] must be an object`);
      out.predicates[normalize(text)] = parsePredicateSettings(item as Record<string, unknown>, `refinements.predicates[${JSON.stringify(text)}]`, []);
    }
  }
  if (raw.services !== undefined) {
    const services = raw.services as Record<string, unknown>;
    if (!services || typeof services !== 'object' || Array.isArray(services)) throw new TypeError('refinements.services must be an object');
    out.services = {};
    for (const [name, type] of Object.entries(services)) {
      if (typeof type !== 'string' || !type) throw new TypeError(`refinements.services.${name} must be type text`);
      out.services[name] = type;
    }
  }
  return out;
}

export const normalizePredicate = (predicate: string): string => predicate.replace(/\s+/g, ' ').trim();
const normalize = normalizePredicate;

