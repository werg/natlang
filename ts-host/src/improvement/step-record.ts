/**
 * `natlang.improvement-step/1` (plans/neuralese/LEARNING_CONTINUUM.md §8): one record per operator application —
 * a crisp search, a gradient run on a soft value or adapter, a learned updater's proposal. Records are the data spine
 * of learned improvers (§9) and the reward-blind improver (§10): `before`, `view` and `proposal` are what an operator
 * may learn to produce; `outcome` is host-only and never part of any view (`stepView` drops it).
 *
 * Records are content-addressed: `id` is the SHA-256 of the canonical record without `id`, so a converted record and
 * a directly written one of the same step agree.
 */
import { createHash } from 'node:crypto';

export const IMPROVEMENT_STEP_SCHEMA = 'natlang.improvement-step/1';

/** What the operator saw (§10.1): everything, observations without rewards, or observations without any outcome. */
export const VISIBILITY_CLASSES = ['full', 'reward-blind-observations', 'reward-blind-strict'] as const;
export type Visibility = typeof VISIBILITY_CLASSES[number];

/** Artifact kinds (§3). Crisp artifacts are named by the digest of their source, soft ones by block content ID. */
export const ARTIFACT_KINDS = ['crisp-source', 'crisp-skill', 'instruction', 'soft-skill', 'soft-value', 'adapter',
  'adapter-code', 'program', 'system-prompt'] as const;
export type ArtifactKind = typeof ARTIFACT_KINDS[number];

/** Signal regimes (§4) and operator families (§7.1). */
export const REGIMES = ['supervised', 'reinforcement', 'conditioned-distillation', 'self-distillation', 'search',
  'reward-blind', 'none'] as const;
export type Regime = typeof REGIMES[number];

export type ArtifactRef = {
  readonly kind: ArtifactKind;
  /** Content ID (block ID, source digest). `null` only in records converted from runs that did not keep it. */
  readonly id: string | null;
  /** Its place in the program: a skill name, a context key, a function. */
  readonly role?: string;
};

export type Gain = {
  readonly before: number | null;
  readonly after: number | null;
  /** after − before, in the family's quality unit. */
  readonly effect: number | null;
  readonly cases?: number;
  readonly wins?: number;
  readonly losses?: number;
  readonly ties?: number;
};

export type ImprovementStep = {
  readonly schema: typeof IMPROVEMENT_STEP_SCHEMA;
  readonly id: string;
  readonly episode: { readonly id: string; readonly family: string; readonly split?: string };
  /** Facet tags (§9.2): skill type, failure kind, task family group. */
  readonly facets: readonly string[];
  readonly before: readonly ArtifactRef[];
  readonly operator: {
    readonly kind: string;
    readonly version: string;
    readonly regime: Regime;
    readonly hyper?: Record<string, unknown>;
    /** Content ID of a learned operator's own context (its skills, soft blocks, adapter). */
    readonly context?: string | null;
    /** The model that ran the operator, if any. */
    readonly model?: string | null;
  };
  readonly view: { readonly visibility: Visibility; readonly evidence: readonly string[] };
  readonly proposal: { readonly deltas: readonly { readonly artifact: ArtifactRef; readonly delta: string | null; readonly scale: number }[] };
  readonly after: readonly ArtifactRef[];
  /** Host-only. */
  readonly outcome: {
    readonly disposition?: string;
    readonly support?: Gain | null;
    readonly query?: Gain | null;
    readonly transfer?: Gain | null;
    readonly compute?: { readonly model_calls?: number; readonly operator_requests?: number; readonly gradient_steps?: number;
      readonly seconds?: number; readonly tokens?: number };
  } | null;
  readonly trajectory: { readonly id: string; readonly step: number };
  /** Where a converted record came from. */
  readonly provenance?: { readonly converter?: string; readonly file?: string; readonly sha256?: string; readonly legacy?: boolean };
};

const canonical = (value: unknown): unknown => Array.isArray(value) ? value.map(canonical) :
  value && typeof value === 'object' ? Object.fromEntries(Object.entries(value as Record<string, unknown>)
    .filter(([, item]) => item !== undefined).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, item]) => [key, canonical(item)])) : value;

export const digest = (value: unknown): string =>
  createHash('sha256').update(typeof value === 'string' ? value : JSON.stringify(canonical(value))).digest('hex');

/** A record with its schema and content ID filled in; throws if it does not validate. */
export function improvementStep(fields: Omit<ImprovementStep, 'schema' | 'id'>): ImprovementStep {
  const body = { schema: IMPROVEMENT_STEP_SCHEMA, ...fields } as Omit<ImprovementStep, 'id'>;
  const record = { ...body, id: 'step-' + digest(body).slice(0, 32) } as ImprovementStep;
  const errors = validateImprovementStep(record);
  if (errors.length) throw new Error(`invalid improvement step: ${errors.join('; ')}`);
  return record;
}

/** What an operator may be trained to see and produce: the record without its outcome. */
export function stepView(record: ImprovementStep): Omit<ImprovementStep, 'outcome'> {
  const { outcome: _outcome, ...view } = record;
  return view;
}

export function validateImprovementStep(value: unknown): string[] {
  const errors: string[] = [];
  const record = value as Partial<ImprovementStep> | null;
  if (!record || typeof record !== 'object') return ['not an object'];
  if (record.schema !== IMPROVEMENT_STEP_SCHEMA) errors.push(`schema must be ${IMPROVEMENT_STEP_SCHEMA}`);
  if (typeof record.id !== 'string' || !record.id.startsWith('step-')) errors.push('id must be a step- content ID');
  else {
    const { id, ...body } = record;
    if (id !== 'step-' + digest(body).slice(0, 32)) errors.push('id does not match the record content');
  }
  if (!record.episode?.id || !record.episode?.family) errors.push('episode needs id and family');
  if (!Array.isArray(record.facets) || record.facets.some(item => typeof item !== 'string')) errors.push('facets must be strings');
  const legacy = !!record.provenance?.legacy;
  const refs = (where: string, list: unknown) => {
    if (!Array.isArray(list)) { errors.push(`${where} must be a list`); return; }
    for (const [index, ref] of (list as ArtifactRef[]).entries()) {
      if (!ARTIFACT_KINDS.includes(ref?.kind)) errors.push(`${where}[${index}].kind must be one of ${ARTIFACT_KINDS.join(', ')}`);
      if (!(typeof ref?.id === 'string' && ref.id) && !(ref?.id === null && legacy))
        errors.push(`${where}[${index}].id must be a content ID (null only in legacy conversions)`);
    }
  };
  refs('before', record.before);
  refs('after', record.after);
  const operator = record.operator;
  if (!operator?.kind || !operator?.version) errors.push('operator needs kind and version');
  if (!REGIMES.includes(operator?.regime as Regime)) errors.push(`operator.regime must be one of ${REGIMES.join(', ')}`);
  if (!VISIBILITY_CLASSES.includes(record.view?.visibility as Visibility)) errors.push(`view.visibility must be one of ${VISIBILITY_CLASSES.join(', ')}`);
  if (!Array.isArray(record.view?.evidence)) errors.push('view.evidence must list content IDs');
  if (!Array.isArray(record.proposal?.deltas)) errors.push('proposal.deltas must be a list');
  else refs('proposal.deltas[].artifact', record.proposal.deltas.map(item => item?.artifact));
  if (record.outcome !== null && (typeof record.outcome !== 'object' || record.outcome === undefined)) errors.push('outcome must be an object or null');
  if (!record.trajectory?.id || !Number.isSafeInteger(record.trajectory?.step) || record.trajectory.step < 0)
    errors.push('trajectory needs id and a nonnegative step');
  return errors;
}

/** The gain of a paired evaluation from `SourceEvaluator.confirmQuality` (crisp authoring results). */
export function pairedGain(paired: { baseline?: { quality?: number; total?: number }; selected?: { quality?: number };
  effect?: number; wins?: number; losses?: number; ties?: number } | null | undefined): Gain | null {
  if (!paired) return null;
  return { before: paired.baseline?.quality ?? null, after: paired.selected?.quality ?? null, effect: paired.effect ?? null,
    cases: paired.baseline?.total, wins: paired.wins, losses: paired.losses, ties: paired.ties };
}

/** The step of one crisp authoring result (`natlang.skill-authoring-trajectory/1`), or null when the run failed
 * before a baseline existed (no operator was applied). */
export function authoringStep(result: Record<string, any>, provenance?: ImprovementStep['provenance']): ImprovementStep | null {
  if (!result.baseline || !result.family) return null;
  const search = result.search ?? {}, definition = result.searchDefinition ?? {};
  const calls = (side: any) => (side?.baseline?.modelCalls ?? 0) + (side?.selected?.modelCalls ?? 0);
  const moved = result.disposition === 'evaluated' || result.disposition === 'not-promoted';
  const support = search.validation && search.validation.quality != null && search.baseline?.quality != null
    ? { before: search.baseline.quality, after: search.validation.quality, effect: search.validation.quality - search.baseline.quality } : null;
  return improvementStep({
    episode: { id: result.episode, family: result.family, split: result.split },
    facets: [`family:${result.family}`, 'operator:crisp-skill-search'],
    before: [{ kind: 'crisp-source', id: result.baseline }],
    operator: { kind: 'crisp-skill-search', version: String(definition.version ?? 'unknown'), regime: 'search',
      hyper: { policy: definition.policy ?? null, budget: definition.budget ?? null, seed: definition.seed ?? null },
      context: null, model: typeof result.author_identity === 'string' ? result.author_identity : null },
    // The author sees the support cases and its own experiments: rewards on support are part of its view.
    view: { visibility: 'full', evidence: [result.evaluation_ticket?.id, definition.authoredDigest].filter(Boolean) },
    proposal: { deltas: result.selected && result.selected !== result.baseline ?
      [{ artifact: { kind: 'crisp-source', id: result.selected }, delta: null, scale: 1 }] : [] },
    after: [{ kind: 'crisp-source', id: moved && result.selected ? result.selected : result.baseline }],
    outcome: { disposition: result.disposition, support, query: pairedGain(result.query), transfer: pairedGain(result.transfer),
      compute: { operator_requests: result.authorExchanges?.length ?? 0, model_calls: calls(result.query) + calls(result.transfer) } },
    trajectory: { id: `authoring:${result.identity ?? result.episode}`, step: 0 },
    ...(provenance ? { provenance } : {}),
  });
}
