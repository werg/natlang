// Programs the TypeScript checker must reject, given spec/neuralese.d.ts. Each line under
// `@ts-expect-error` is one rule. Rules TypeScript cannot express are natlang compiler
// diagnostics, covered by conformance/neuralese/cases/compile-errors.yaml.
import { read, type Neuralese } from 'natlang:neuralese';

type Summary = { headline: string; points: string[] };
type Ticket = { id: string };
type Label = 'urgent' | 'normal';

declare const note: Neuralese<Summary>;
declare const summary: Summary;
declare const other: Neuralese<Summary, 'nd:natlang@2'>;
declare const triage: Neuralese<(t: Ticket) => Promise<Label>>;

export async function errors() {
  // neuralese-opaque-access: no fields.
  // @ts-expect-error
  note.headline;

  // neuralese-opaque-access: no arithmetic.
  // @ts-expect-error
  const n = note + 1;

  // A Neuralese<T> is not a T.
  // @ts-expect-error
  const s: Summary = note;

  // A T is not a Neuralese<T>.
  // @ts-expect-error
  const v: Neuralese<Summary> = summary;

  // neuralese-nested: Neuralese<Neuralese<T>> is never.
  // @ts-expect-error
  const nested: Neuralese<Neuralese<Summary>> = note;

  // neuralese-dialect-mismatch.
  // @ts-expect-error
  const mixed: Neuralese<Summary> = other;

  // A soft function keeps its signature.
  // @ts-expect-error
  await triage('not a ticket');

  // read returns the declared T.
  // @ts-expect-error
  const wrong: Ticket = await read(note);

  // A soft value of a non-function type is not callable.
  // @ts-expect-error
  await note();

  return { n, s, v, nested, mixed, wrong };
}
