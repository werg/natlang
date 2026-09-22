import { crisp, natural } from './builders.mjs';

const types = `type State = { count: number, step: number };
type UiEvent = { id: string, kind: string, value?: string };
type Action = { kind: string, value?: string, from?: string };
type Node = { tag: string, text?: string, value?: string, id?: string, label?: string, disabled?: boolean, action?: Action, children?: Node[] };`;
const reducer = `/*---
args:
  state: State
  event: UiEvent
returns: State
engine: typescript-host
---*/
const { state, event } = args;
if (event.kind === "reset") return { ...state, count: 0 };
if (event.kind === "step") {
  const step = Number(event.value);
  return { ...state, step: Number.isFinite(step) ? Math.max(1, Math.min(100, step)) : state.step };
}
const direction = event.kind === "increase" ? 1 : event.kind === "decrease" ? -1 : 0;
return { ...state, count: state.count + direction * state.step };
`;
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
  code: `const view = ${counterView.toString()};\nreturn view(args.state);`,
  inputs: { state: { count: 3, step: 1 } }, expected: counterView({ count: 3, step: 1 }) });
counter.guide = 'Click Increase, edit ui/view.ts, or scrub the timeline. Live edit keeps the current count.';
const modelCounter = natural({ id: 'natural-interface', name: 'Generated counter UI', category: 'Live interfaces', level: 'Intermediate',
  description: 'Let natlang generate an interactive interface from typed state and plain-language design instructions.',
  concepts: ['live UI', 'generated views', 'typed output'], root: 'ui/view.nl',
  files: { 'ui/types.ts': types, 'ui/reduce.ts': reducer }, inputs: { state: { count: 3, step: 1 } },
  source: `---
args:
  state: State
returns: Node
---
Create a counter interface as a Node tree. Write it to return.
Use a section containing a heading, an output showing args/state/count,
and three buttons labelled Decrease, Increase, and Reset.
Their action kinds must be decrease, increase, and reset respectively.
Add a short paragraph explaining that each click changes the count by args/state/step.
Use only section, div, h2, p, output, and button tags.
Do not emit HTML, CSS, or JavaScript. Return structured Node data.` });
modelCounter.guide = 'The model writes the view; the exact reducer handles clicks. Try asking for a different heading or explanation in ui/view.nl, then Apply & run. Each interaction generates a fresh view.';
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
  files: { 'ui/types.ts': quoteTypes, 'ui/reduce.ts': `/*---
args:
  state: State
  event: UiEvent
returns: State
engine: typescript-host
---*/
if (args.event.kind === "billing") return { ...args.state, annual: !args.state.annual };
const seats = Number(args.event.value);
return { ...args.state, seats: Number.isFinite(seats) ? Math.max(1, Math.min(1000, Math.round(seats))) : args.state.seats };` },
  code: `const view = ${quoteView.toString()};\nreturn view(args.state);`,
  inputs: { state: { seats: 5, annual: false } }, expected: quoteView({ seats: 5, annual: false }) });
quote.guide = 'Change the seats and switch billing. Then change the prices in ui/view.ts with live editing on. The state stays put while the calculation changes.';
const modelQuote = natural({ id: 'natural-pricing-ui', name: 'Generated pricing UI', category: 'Live interfaces', level: 'Intermediate',
  description: 'Generate a pricing interface from state, with exact billing logic handled by a typed reducer.',
  concepts: ['live UI', 'generated views', 'typed state'], root: 'ui/view.nl',
  files: { 'ui/types.ts': quoteTypes, 'ui/reduce.ts': quote.files['ui/reduce.ts'] },
  inputs: { state: { seats: 5, annual: false } },
  source: `---
args:
  state: State
returns: Node
---
Create a pricing interface as a Node tree and write it to return.
Show a heading, an output with the monthly total in euros, an input labelled Team seats,
and a button to switch between monthly and annual billing.
The monthly price is 10 euros per seat, or 8 euros per seat for annual billing.
The Team seats input must send action kind seats with its entered value.
The billing button must send action kind billing.
Use only section, div, h2, p, output, input, and button tags.
Return structured Node data, not HTML, CSS, or JavaScript.` });
modelQuote.guide = 'Edit ui/view.nl to change what the interface shows. The typed reducer in ui/reduce.ts keeps billing changes exact.';
export const interfaceExamples = [counter, quote, modelCounter, modelQuote];
