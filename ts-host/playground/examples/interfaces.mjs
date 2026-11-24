import { crisp, typescript } from './builders.mjs';

/** The view function's source with its parameter and result typed. */
const typedView = view => view.toString().replace(/^state =>/, '(state: State): Node =>');

const types = `type State = { count: number, step: number };
type UiEvent = { id: string, kind: string, value?: string };
type Action = { kind: string, value?: string, from?: string };
type Node = { tag: string, text?: string, value?: string, id?: string, label?: string, disabled?: boolean, action?: Action, children?: Node[] };`;
const reducer = typescript('reduce', { state: 'State', event: 'UiEvent' }, 'State', `
if (event.kind === "reset") return { ...state, count: 0 };
if (event.kind === "step") {
  const step = Number(event.value);
  return { ...state, step: Number.isFinite(step) ? Math.max(1, Math.min(100, step)) : state.step };
}
const direction = event.kind === "increase" ? 1 : event.kind === "decrease" ? -1 : 0;
return { ...state, count: state.count + direction * state.step };
`, types);
const counterView = state => ({ tag: 'section', children: [
  { tag: 'h2', text: 'Counter' },
  { tag: 'output', text: String(state.count) },
  { tag: 'div', children: [
    { tag: 'button', text: '− Decrease', action: { kind: 'decrease' } },
    { tag: 'button', text: '+ Increase', action: { kind: 'increase' } },
    { tag: 'button', text: 'Reset', action: { kind: 'reset' } },
  ] },
  { tag: 'input', label: 'Step size', value: String(state.step), action: { kind: 'step' } },
  { tag: 'p', text: 'Step: ' + state.step },
] });
const counter = crisp({ id: 'living-counter', name: 'Counter UI', category: 'Live interfaces', level: 'Beginner',
  description: 'Counter with editable view, state updates, and recorded history.',
  concepts: ['live UI', 'events', 'state', 'time travel'], root: 'ui/view.ts', args: { state: 'State' }, returns: 'Node',
  files: { 'ui/types.ts': types, 'ui/reduce.ts': reducer },
  code: `const view = ${typedView(counterView)};\nreturn view(state);`,
  inputs: { state: { count: 3, step: 1 } }, expected: counterView({ count: 3, step: 1 }) });
counter.guide = 'Click Increase, edit ui/view.ts, or scrub the timeline. Live edit keeps the current count.';

const semanticCounterTypes = `export type State = { count: number; step: number; note: string };
export type UiEvent = { id: string; kind: string; value?: string };
export type Action = { kind: string; value?: string; from?: string };
export type Node = { tag: string; text?: string; value?: string; id?: string; label?: string; disabled?: boolean; action?: Action; children?: Node[] };
export type Status = 'on-track' | 'attention' | 'complete';
export type Caption = Is<string, 'one concise, encouraging status line under 12 words'>;
export type ViewPlan = { status: Status; confidence: number; caption: Caption };
`;
const semanticCounterReducer = typescript('reduce', { state: 'State', event: 'UiEvent' }, 'State', `
if (event.kind === 'reset') return { ...state, count: 0 };
if (event.kind === 'note') return { ...state, note: String(event.value ?? '') };
if (event.kind === 'step') {
  const step = Number(event.value);
  return { ...state, step: Number.isFinite(step) ? Math.max(1, Math.min(100, step)) : state.step };
}
const direction = event.kind === 'increase' ? 1 : event.kind === 'decrease' ? -1 : 0;
return { ...state, count: state.count + direction * state.step };
`, semanticCounterTypes);
const modelCounter = {
  id: 'natural-interface',
  name: 'Progress tracker UI',
  category: 'Live interfaces',
  level: 'Intermediate',
  description: 'Exact TypeScript view rendering from a typed view plan backed by a calibrated decision leaf and refined caption.',
  concepts: ['live UI', 'readout: decision', 'Untrusted', 'Is<T, P>'],
  modelRequired: true,
  root: 'ui/view.ts',
  files: {
    'ui/types.ts': semanticCounterTypes,
    'ui/reduce.ts': semanticCounterReducer,
    'ui/status.nl': `---
description: Judge whether current progress is on track, needs attention, or complete against the goal note.
args:
  count: number
  step: number
  note: Untrusted<string>
returns: Status
readout: decision
---
Judge progress from count (incremented by step) against note: choose "complete" when the goal in note is met,
"attention" when count is negative or stalled behind the goal, otherwise "on-track".
`,
    'ui/view.ts': `import { decide, nl, untrusted } from '@natlang/browser';
import status from './status.nl';
import type { Caption, Node, State, ViewPlan } from './types.js';

/** Exact deterministic renderer: builds the UI tree from typed state and a semantic ViewPlan. */
function renderPlan(state: State, plan: ViewPlan): Node {
  return {
    tag: 'section',
    children: [
      { tag: 'h2', text: \`Progress · \${plan.status} (\${Math.round(plan.confidence * 100)}%)\` },
      { tag: 'output', text: String(state.count) },
      { tag: 'p', text: plan.caption },
      { tag: 'div', children: [
        { tag: 'button', text: '− Decrease', action: { kind: 'decrease' } },
        { tag: 'button', text: '+ Increase', action: { kind: 'increase' } },
        { tag: 'button', text: 'Reset', action: { kind: 'reset' } },
      ] },
      { tag: 'input', label: 'Step size', value: String(state.step), action: { kind: 'step' } },
      { tag: 'input', label: 'Goal note', value: state.note, action: { kind: 'note' } },
    ],
  };
}

export async function main(input: { state: State }): Promise<Node> {
  const { state } = input;
  const safeNote = untrusted(state.note, 'goal note');
  const draftCaption = nl<(count: number, step: number, note: Untrusted<string>) => Caption>\`Write one short status line describing count (step size step) relative to note.\`;
  const [judged, caption] = await Promise.all([
    decide(status, state.count, state.step, safeNote),
    draftCaption(state.count, state.step, safeNote),
  ]);
  return renderPlan(state, { status: judged.value, confidence: judged.confidence, caption });
}
`,
  },
  inputs: { state: { count: 3, step: 1, note: 'Reach 5 completed reviews today.' } },
  guide: 'TypeScript renders the exact DOM tree and handles every click; ui/status.nl scores the progress state and an inline nl writes the refined caption.',
};

const quoteTypes = `type State = { seats: number, annual: boolean };
type UiEvent = { id: string, kind: string, value?: string };
type Action = { kind: string, value?: string };
type Node = { tag: string, text?: string, value?: string, label?: string, action?: Action, children?: Node[] };`;
const quoteView = state => {
  const price = state.annual ? 8 : 10;
  return { tag: 'section', children: [
    { tag: 'h2', text: 'Pricing' },
    { tag: 'output', text: '€' + state.seats * price + ' / month' },
    { tag: 'input', label: 'Team seats', value: String(state.seats), action: { kind: 'seats' } },
    { tag: 'p', text: state.seats + ' seats × €' + price + ' per seat' },
    { tag: 'button', text: state.annual ? 'Annual billing · switch to monthly' : 'Monthly billing · save 20% annually', action: { kind: 'billing' } },
    { tag: 'p', text: state.annual ? '€' + state.seats * price * 12 + ' billed yearly.' : 'Billed monthly.' },
  ] };
};
const quote = crisp({ id: 'reactive-pricing', name: 'Pricing calculator', category: 'Live interfaces', level: 'Beginner',
  description: 'Change seats and billing to update the total, or edit the calculation.',
  concepts: ['live UI', 'derived values', 'interactive inputs'], root: 'ui/view.ts', args: { state: 'State' }, returns: 'Node',
  files: { 'ui/types.ts': quoteTypes, 'ui/reduce.ts': typescript('reduce', { state: 'State', event: 'UiEvent' }, 'State', `
if (event.kind === "billing") return { ...state, annual: !state.annual };
const seats = Number(event.value);
return { ...state, seats: Number.isFinite(seats) ? Math.max(1, Math.min(1000, Math.round(seats))) : state.seats };`, quoteTypes) },
  code: `const view = ${typedView(quoteView)};\nreturn view(state);`,
  inputs: { state: { seats: 5, annual: false } }, expected: quoteView({ seats: 5, annual: false }) });
quote.guide = 'Change the seats and switch billing. Then change the prices in ui/view.ts with live editing on. The state stays put while the calculation changes.';

const semanticPricingTypes = `export type State = { seats: number; annual: boolean; customerNote: string };
export type UiEvent = { id: string; kind: string; value?: string };
export type Action = { kind: string; value?: string };
export type Node = { tag: string; text?: string; value?: string; label?: string; action?: Action; children?: Node[] };
export type Tier = 'starter' | 'team' | 'enterprise';
export type Callout = Is<string, 'one concise, honest pricing recommendation under 18 words'>;
export type PricingPlan = { tier: Tier; confidence: number; callout: Callout };
`;
const semanticPricingReducer = typescript('reduce', { state: 'State', event: 'UiEvent' }, 'State', `
if (event.kind === 'billing') return { ...state, annual: !state.annual };
if (event.kind === 'note') return { ...state, customerNote: String(event.value ?? '') };
const seats = Number(event.value);
return { ...state, seats: Number.isFinite(seats) ? Math.max(1, Math.min(1000, Math.round(seats))) : state.seats };
`, semanticPricingTypes);
const modelQuote = {
  id: 'natural-pricing-ui',
  name: 'Pricing advisor UI',
  category: 'Live interfaces',
  level: 'Intermediate',
  description: 'Exact billing arithmetic and deterministic UI rendering combined with a calibrated tier decision and refined callout.',
  concepts: ['live UI', 'readout: decision', 'Untrusted', 'Is<T, P>'],
  modelRequired: true,
  root: 'ui/view.ts',
  files: {
    'ui/types.ts': semanticPricingTypes,
    'ui/reduce.ts': semanticPricingReducer,
    'ui/tier.nl': `---
description: Recommend the pricing tier that fits the seat count, billing cadence, and customer context.
args:
  seats: number
  annual: boolean
  note: Untrusted<string>
returns: Tier
readout: decision
---
Choose "enterprise" when note mentions SSO, audit compliance, or large-company procurement; choose "team" for
collaborative groups of 5 or more seats; otherwise choose "starter".
`,
    'ui/view.ts': `import { decide, nl, untrusted } from '@natlang/browser';
import tier from './tier.nl';
import type { Callout, Node, PricingPlan, State } from './types.js';

/** Exact deterministic renderer: prices are computed in TypeScript and paired with the typed PricingPlan. */
function renderPricing(state: State, plan: PricingPlan): Node {
  const perSeat = state.annual ? 8 : 10;
  const monthly = state.seats * perSeat;
  const yearly = monthly * 12;
  return {
    tag: 'section',
    children: [
      { tag: 'h2', text: \`Recommended: \${plan.tier} (\${Math.round(plan.confidence * 100)}%)\` },
      { tag: 'output', text: \`€\${monthly} / month\` },
      { tag: 'p', text: \`\${state.seats} seats × €\${perSeat} per seat · \${state.annual ? \`€\${yearly} billed yearly\` : 'billed monthly'}\` },
      { tag: 'p', text: plan.callout },
      { tag: 'input', label: 'Team seats', value: String(state.seats), action: { kind: 'seats' } },
      { tag: 'input', label: 'Customer note', value: state.customerNote, action: { kind: 'note' } },
      { tag: 'button', text: state.annual ? 'Annual billing · switch to monthly' : 'Monthly billing · save 20% annually', action: { kind: 'billing' } },
    ],
  };
}

export async function main(input: { state: State }): Promise<Node> {
  const { state } = input;
  const safeNote = untrusted(state.customerNote, 'customer note');
  const writeCallout = nl<(seats: number, annual: boolean, tier: string, note: Untrusted<string>) => Callout>\`Write one short pricing recommendation for seats seats (annual billing: annual) on the tier plan given note.\`;
  const judged = await decide(tier, state.seats, state.annual, safeNote);
  const callout = await writeCallout(state.seats, state.annual, judged.value, safeNote);
  return renderPricing(state, { tier: judged.value, confidence: judged.confidence, callout });
}
`,
  },
  inputs: { state: { seats: 12, annual: false, customerNote: 'Growing product team looking to save on yearly renewals.' } },
  guide: 'TypeScript computes every euro total and builds the UI tree; ui/tier.nl scores the recommended tier and an inline nl writes the refined callout.',
};

export const interfaceExamples = [counter, quote, modelCounter, modelQuote];
